import os
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np

# Define directories
RECORDED_DIR = "recorded_events"
ROOSTER_DIR = "rooster"
NOISE_DIR = "noise"

# Ensure output directories exist
os.makedirs(ROOSTER_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)

# Set Jabra as the output device
sd.default.device = 1  # Jabra SPEAK 510 USB

def play_audio(file_path):
    """Plays a .wav file, resampling to 48 kHz if needed."""
    try:
        data, samplerate = sf.read(file_path)

        # Jabra forces 48 kHz playback, so resample if needed
        target_rate = 48000
        if samplerate != target_rate:
            print(f"Resampling from {samplerate} Hz → {target_rate} Hz for Jabra")
            data = np.interp(
                np.linspace(0, len(data), int(len(data) * target_rate / samplerate)),
                np.arange(len(data)),
                data
            )

        print(f"Playing {file_path} at {target_rate} Hz")
        sd.play(data, samplerate=target_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

print("\n🔊 Audio classification started.")
print("Press 'r' to label as rooster, 'n' for noise, or 'q' to quit.")

for filename in sorted(os.listdir(RECORDED_DIR)):
    if filename.endswith(".wav"):
        file_path = os.path.join(RECORDED_DIR, filename)

        print(f"\n🎵 Playing: {filename}")
        play_audio(file_path)

        while True:
            command = input("Enter classification (r=rooster, n=noise, q=quit): ").strip().lower()
            if command == 'r':
                shutil.move(file_path, os.path.join(ROOSTER_DIR, filename))
                print(f"✅ Moved to: {ROOSTER_DIR}")
                break
            elif command == 'n':
                shutil.move(file_path, os.path.join(NOISE_DIR, filename))
                print(f"✅ Moved to: {NOISE_DIR}")
                break
            elif command == 'q':
                print("🚪 Exiting classification.")
                exit()
            else:
                print("❌ Invalid input. Use 'r' for rooster, 'n' for noise, or 'q' to quit.")

print("🎉 All recordings sorted successfully!")
