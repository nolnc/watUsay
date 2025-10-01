import argparse
import torch
import colorama
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import sys
from contextlib import contextmanager

# Configuration
SAMPLE_RATE = 16000
CHUNK_FRAMES = 4096 # Number of frames to read at a time

def clear_console_lines(n=1):
    """Clears the last N lines in the console."""
    for _ in range(n):
        # Move cursor up one line
        sys.stdout.write('\x1b[1A')
        # Clear the entire line
        sys.stdout.write('\x1b[2K')

def record_audio(device_index, q, stop_event):
    """
    This function runs in a separate thread to continuously record audio
    using a non-blocking callback.
    """
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(f"Audio Status Warning: {status}", file=sys.stderr)
        q.put(indata.copy())

    try:
        # The InputStream is opened with a callback function.
        # It runs in the background and calls `callback` for each new audio chunk.
        with sd.InputStream(samplerate=SAMPLE_RATE, device=device_index, channels=1, callback=callback, blocksize=CHUNK_FRAMES):
            stop_event.wait()  # Keep the stream open until the main thread signals to stop.
    except Exception as e:
        print(f"Error in recording thread: {e}")


@contextmanager
def recording_thread(device_index, audio_queue):
    """A context manager to handle the recording thread's lifecycle."""
    stop_event = threading.Event()
    recorder = threading.Thread(
        target=record_audio,
        args=(device_index, audio_queue, stop_event)
    )
    recorder.start()
    try:
        yield
    finally:
        stop_event.set()
        recorder.join()


def select_device():
    """Interactively lists and selects an audio device."""
    devices = sd.query_devices()
    input_devices = []
    suggested_devices = []
    loopback_keywords = ['stereo mix', 'wave out', 'what u hear', 'loopback', 'monitor']

    # --- Find all input devices and identify potential loopback devices ---
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
            is_loopback = any(keyword in device['name'].lower() for keyword in loopback_keywords)
            if is_loopback:
                suggested_devices.append((i, device['name']))
                try:
                    # Check if the device supports the required sample rate and channels
                    sd.check_input_settings(device=i, samplerate=SAMPLE_RATE, channels=1)
                except sd.PortAudioError:
                    # Device is not compatible or enabled
                    pass

    # --- Display device lists to the user ---
    if suggested_devices:
        print("\n--- Suggested Loopback Devices ---")
        for i, name in suggested_devices:
            print(f"-> {i}: {name}")

    if not input_devices:
        print("Error: No input devices found.", file=sys.stderr)
        return None

    print("\n--- All Available Input Devices ---")
    for i, name in input_devices:
        print(f"  {i}: {name}")

    try:
        device_index = int(input("\nPlease enter the device index to use: "))
        device_info = sd.query_devices(device_index)
        if device_info['max_input_channels'] == 0:
            print(f"Error: Device {device_index} is not an input device.", file=sys.stderr)
            return None
        return device_index
    except (ValueError, TypeError, IndexError):
        print("Error: Invalid device index. Please enter a valid number from the list.", file=sys.stderr)
        return None


def main():
    # Initialize colorama to make ANSI escape codes work on Windows
    colorama.init()

    parser = argparse.ArgumentParser(description="Live audio translation using Whisper.")
    parser.add_argument("--model", default="base", help="Whisper model to use (e.g., tiny, base, small, medium).")
    parser.add_argument("--buffer_seconds", type=int, default=5, help="Seconds of audio to buffer before translating.")
    parser.add_argument("--language", default=None, help="Language of the audio to be translated (e.g., 'zh', 'ja', 'de'). Default is auto-detect.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task to perform: 'transcribe' (X->X) or 'translate' (X->English).")
    parser.add_argument("--energy_threshold", type=float, default=0.005, help="Energy threshold for detecting speech. Lower values are more sensitive.")
    args = parser.parse_args()

    # --- Device Selection ---
    device_index = select_device()
    if device_index is None:
        return

    # --- Threading Setup ---
    audio_queue = queue.Queue()
    buffer_frames = SAMPLE_RATE * args.buffer_seconds

    print(f"\nUsing device: {sd.query_devices(device_index)['name']}")
    print(f"Loading Whisper '{args.model}' model for '{args.task}' task...")
    model = whisper.load_model(args.model)
    print("Model loaded. Recording... Press Ctrl+C to stop.")

    audio_buffer = np.array([], dtype=np.float32)

    try:
        with recording_thread(device_index, audio_queue):
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=0.5)

                    current_audio = audio_chunk.flatten()
                    audio_buffer = np.concatenate([audio_buffer, current_audio])

                    buffered_seconds = len(audio_buffer) / SAMPLE_RATE
                    print(f"Buffering: {buffered_seconds:.2f}s / {args.buffer_seconds:.2f}s", end='\r', flush=True)

                    if len(audio_buffer) >= buffer_frames:
                        # Overwrite "Buffering..." line with processing status
                        print(f"Buffer full. Analyzing...", end='\r', flush=True)

                        # Simple VAD: Check the energy of the buffer
                        rms = np.sqrt(np.mean(audio_buffer**2))
                        if rms > args.energy_threshold:
                            print(f"Buffer full. {args.task.capitalize()}...")

                            audio_to_process = whisper.audio.pad_or_trim(audio_buffer)
                            mel = whisper.audio.log_mel_spectrogram(audio_to_process).to(model.device)

                            options = whisper.DecodingOptions(task=args.task, language=args.language, without_timestamps=True, fp16=torch.cuda.is_available())
                            result = whisper.decode(model, mel, options)

                            if result.text.strip():
                                clear_console_lines(1)
                                print(f"-> {result.text.strip()}")

                        # Clear the buffer to process the next distinct chunk of audio
                        audio_buffer = np.array([], dtype=np.float32)

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\nStopping...")
        # Process any remaining audio in the buffer before exiting
        if len(audio_buffer) > SAMPLE_RATE: # Process if more than 1s of audio
            print("Processing remaining audio...", end="", flush=True)
            audio_to_process = whisper.audio.pad_or_trim(audio_buffer)
            mel = whisper.audio.log_mel_spectrogram(audio_to_process).to(model.device)
            options = whisper.DecodingOptions(task=args.task, language=args.language, without_timestamps=True, fp16=torch.cuda.is_available())
            result = whisper.decode(model, mel, options)
            print("Done.")
            clear_console_lines(2)
            if result.text.strip():
                print(f"-> {result.text.strip()}")
        print("Script stopped.")

if __name__ == "__main__":
    main()