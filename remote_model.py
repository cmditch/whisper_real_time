#! python3.7

import argparse
import io
import os
import wave
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

import numpy as np
import speech_recognition as sr
from openai import OpenAI


class WhisperTranscriber:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(api_key=args.api_key)
        self.data_queue = Queue()
        self.chunk_bytes = bytes()
        self.transcription = [""]
        self.phrase_time = None
        self.last_api_call = None

        # Load existing transcription if file exists
        self.original_line_count = 0
        self._load_existing_transcription()

        # Set up audio recorder
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        self.recorder.dynamic_energy_threshold = False

        # Set up microphone source
        self.source = self._setup_microphone()

        if self.source:
            with self.source:
                self.recorder.adjust_for_ambient_noise(self.source)

    def _load_existing_transcription(self):
        """Load existing transcription from file if it exists."""
        if not self.args.output_file or not os.path.exists(self.args.output_file):
            return

        try:
            with open(self.args.output_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse existing transcription lines (skip comments and empty lines)
            existing_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    existing_lines.append(line)

            if existing_lines:
                # Replace the default empty transcription with existing content
                self.transcription = existing_lines
                print(
                    f"Loaded {len(existing_lines)} existing transcription lines from {self.args.output_file}"
                )

        except Exception as e:
            print(f"Error loading existing transcription: {e}")

    def _setup_microphone(self):
        """Set up the microphone source based on user selection."""
        mic_name = self.args.mic

        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'  - "{name}"')
            return None
        elif mic_name:
            # Search for the specified microphone
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name.lower() in name.lower():
                    print(f"Using microphone: {name}")
                    return sr.Microphone(sample_rate=16000, device_index=index)

            print(f"Microphone '{mic_name}' not found. Available microphones:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  {name}")
            return None
        else:
            # Use default microphone
            return sr.Microphone(sample_rate=16000)

    def record_callback(self, _, audio: sr.AudioData) -> None:
        """Threaded callback function to receive audio data when recordings finish."""
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def split_text_at_periods(self, text: str, max_length: int = 120) -> list[str]:
        """Split text into chunks at periods, targeting max_length characters."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        remaining_text = text

        while len(remaining_text) > max_length:
            # Find the ideal split point around max_length
            search_start = max(0, max_length - 30)  # Look 30 chars before ideal point
            search_end = min(
                len(remaining_text), max_length + 30
            )  # Look 30 chars after ideal point
            search_section = remaining_text[search_start:search_end]

            # Find all periods in the search section
            period_positions = [
                i for i, char in enumerate(search_section) if char == "."
            ]

            if period_positions:
                # Find the period closest to our target length
                target_pos = max_length - search_start
                best_period = min(period_positions, key=lambda x: abs(x - target_pos))
                split_point = search_start + best_period + 1  # +1 to include the period

                # Extract the chunk and add it
                chunk = remaining_text[:split_point].strip()
                if chunk:
                    chunks.append(chunk)
                remaining_text = remaining_text[split_point:].strip()
            else:
                # No period found, split at max_length at a word boundary
                split_point = max_length
                while split_point > 0 and remaining_text[split_point] != " ":
                    split_point -= 1

                if split_point == 0:  # No space found, just split at max_length
                    split_point = max_length

                chunk = remaining_text[:split_point].strip()
                if chunk:
                    chunks.append(chunk)
                remaining_text = remaining_text[split_point:].strip()

        # Add any remaining text
        if remaining_text.strip():
            chunks.append(remaining_text.strip())

        return chunks

    def audio_to_wav_bytes(self, audio_data: bytes, sample_rate: int = 16000) -> bytes:
        """Convert raw audio data to WAV format bytes."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_np.tobytes())

        wav_buffer.seek(0)
        return wav_buffer.getvalue()

    def transcribe_audio_chunk(self, audio_bytes: bytes) -> str:
        """Send audio chunk to OpenAI Whisper API for transcription."""
        try:
            # Convert to WAV format
            wav_data = self.audio_to_wav_bytes(audio_bytes)

            # Create a temporary file-like object
            audio_file = io.BytesIO(wav_data)
            audio_file.name = "audio.wav"  # API needs a filename

            # Call OpenAI Whisper API
            transcript = self.client.audio.transcriptions.create(
                model=self.args.model, file=audio_file, language=self.args.language
            )

            return transcript.text.strip()

        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def write_transcription_to_file(self) -> None:
        """Write the current transcription to the output file."""
        if not self.args.output_file:
            return

        try:
            # Check if we need to add a restart marker
            file_exists = os.path.exists(self.args.output_file)
            needs_restart_marker = file_exists and not hasattr(
                self, "_restart_marker_added"
            )

            with open(self.args.output_file, "w", encoding="utf-8") as f:
                f.write(
                    f"# Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n\n"
                )

                # Add restart marker if this is a continuation
                if needs_restart_marker:
                    f.write(
                        f"# =================== RESTARTED {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ===================\n\n"
                    )
                    self._restart_marker_added = True

                for line in self.transcription:
                    if line.strip():  # Only write non-empty lines
                        f.write(f"{line}\n")

                f.write(
                    f"\n\n# Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC\n"
                )
        except Exception as e:
            print(f"Error writing to file: {e}")

    def display_transcription(self) -> None:
        """Clear console and display current transcription."""
        os.system("cls" if os.name == "nt" else "clear")
        print("Real-time Whisper API Transcription")
        print("=" * 40)
        for i, line in enumerate(self.transcription):
            if line.strip():  # Only print non-empty lines
                print(f"{i + 1}: {line}")
        print("", end="", flush=True)

    def start_transcription(self):
        """Start the real-time transcription process."""
        if not self.source:
            return

        # Create a background thread that will pass us raw audio bytes
        self.recorder.listen_in_background(
            self.source,
            self.record_callback,
            phrase_time_limit=self.args.record_timeout,
        )

        # Cue the user that we're ready to go
        print("Connected to OpenAI Whisper API. Starting transcription...\n")

        # Write initial empty transcription to file
        if self.args.output_file:
            print(f"Transcription will be saved to: {self.args.output_file}")
            self.write_transcription_to_file()

        while True:
            try:
                now = datetime.utcnow()

                # Pull raw recorded audio from the queue
                if not self.data_queue.empty():
                    phrase_complete = False

                    # If enough time has passed between recordings, consider the phrase complete
                    if self.phrase_time and now - self.phrase_time > timedelta(
                        seconds=self.args.phrase_timeout
                    ):
                        phrase_complete = True

                    # This is the last time we received new audio data from the queue
                    self.phrase_time = now

                    # Combine audio data from queue
                    while not self.data_queue.empty():
                        audio_data = self.data_queue.get()
                        self.chunk_bytes += audio_data

                    # Check if we should send chunk to API
                    should_transcribe = False

                    # Send to API if:
                    # 1. We have a phrase complete and some audio data
                    # 2. We've accumulated enough audio (chunk_duration)
                    # 3. It's been a while since last API call and we have audio
                    # 4. We have audio and there's been silence for flush_timeout seconds

                    if phrase_complete and self.chunk_bytes:
                        should_transcribe = True
                    elif (
                        self.last_api_call is None
                        or now - self.last_api_call
                        > timedelta(seconds=self.args.chunk_duration)
                    ) and self.chunk_bytes:
                        should_transcribe = True

                    if (
                        should_transcribe and len(self.chunk_bytes) > 1000
                    ):  # Minimum audio size
                        print("🎙️  Transcribing...", end="", flush=True)

                        # Transcribe the chunk
                        text = self.transcribe_audio_chunk(self.chunk_bytes)
                        self.last_api_call = now

                        if text:
                            # Combine with existing text if appropriate
                            if self.transcription[-1] and not phrase_complete:
                                combined_text = self.transcription[-1] + " " + text
                                self.transcription.pop()  # Remove the last line
                            else:
                                combined_text = text

                            # Split the text into appropriately sized chunks
                            text_chunks = self.split_text_at_periods(combined_text)

                            # Add all chunks to transcription
                            for chunk in text_chunks:
                                if chunk.strip():
                                    self.transcription.append(chunk.strip())

                        # Clear the chunk after processing
                        self.chunk_bytes = bytes()

                        # Display updated transcription
                        self.display_transcription()

                        # Write transcription to file
                        self.write_transcription_to_file()
                else:
                    # Check if we should flush remaining audio after silence
                    if (
                        self.chunk_bytes
                        and self.phrase_time
                        and now - self.phrase_time
                        > timedelta(seconds=self.args.flush_timeout)
                    ):
                        print("🔄 Flushing remaining audio...", end="", flush=True)

                        # Transcribe the remaining chunk
                        text = self.transcribe_audio_chunk(self.chunk_bytes)
                        self.last_api_call = now

                        if text:
                            # Combine with existing text if appropriate
                            if self.transcription[-1]:
                                combined_text = self.transcription[-1] + " " + text
                                self.transcription.pop()  # Remove the last line
                            else:
                                combined_text = text

                            # Split the text into appropriately sized chunks
                            text_chunks = self.split_text_at_periods(combined_text)

                            # Add all chunks to transcription
                            for chunk in text_chunks:
                                if chunk.strip():
                                    self.transcription.append(chunk.strip())

                        # Clear the chunk after processing
                        self.chunk_bytes = bytes()

                        # Display updated transcription
                        self.display_transcription()

                        # Write transcription to file
                        self.write_transcription_to_file()

                    # Infinite loops are bad for processors, must sleep
                    sleep(0.25)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                sleep(1)

        # Final transcription if there's remaining audio
        if self.chunk_bytes:
            print("\n🎙️  Processing final audio chunk...")
            text = self.transcribe_audio_chunk(self.chunk_bytes)
            if text:
                self.transcription.append(text)

        print("\n\nFinal Transcription:")
        print("=" * 50)
        for i, line in enumerate(self.transcription):
            if line.strip():
                print(f"{i + 1}: {line}")

        # Write final transcription to file
        if self.args.output_file:
            self.write_transcription_to_file()
            print(f"\nTranscription saved to: {self.args.output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument(
        "--model", default="whisper-1", help="Whisper model to use (whisper-1)"
    )
    parser.add_argument(
        "--language", default=None, help="Language code (e.g., 'en', 'es', 'fr')"
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )
    parser.add_argument(
        "--flush_timeout",
        default=2,
        help="How much silence before processing any remaining audio buffer.",
        type=float,
    )
    parser.add_argument(
        "--chunk_duration",
        default=10,
        help="Duration in seconds to accumulate audio before sending to API.",
        type=float,
    )
    parser.add_argument(
        "--mic",
        default=None,
        help="Default microphone name for SpeechRecognition. "
        "Run this with 'list' to view available Microphones.",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="wat.txt",
        help="File path to write transcription output. If not specified, only console output is used.",
        type=str,
    )
    args = parser.parse_args()

    # Create and start the transcriber
    transcriber = WhisperTranscriber(args)
    transcriber.start_transcription()


if __name__ == "__main__":
    main()
