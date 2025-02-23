import os
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np

# Directories:
# RECORDED_DIR should be the local copy of the "recorded_events" folder from the Raspberry Pi.
RECORDED_DIR = "recorded_events"
ROOSTER_DIR = "rooster"
NOISE_DIR = "noise"

# Ensure the output directories exist.
os.makedirs(ROOSTER_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)

def play_audio(file_path):
    """
    Plays a .wav file, resampling to 48 kHz if needed.
    """
    try:
        data, samplerate = sf.read(file_path)
        target_rate = 48000  # Use 48000 Hz as the target sample rate.
        
        # If needed, resample the data.
        if samplerate != target_rate:
            print(f"Resampling from {samplerate} Hz → {target_rate} Hz")
            data = np.interp(
                np.linspace(0, len(data), int(len(data) * target_rate / samplerate)),
                np.arange(len(data)),
                data
            )
        
        print(f"Playing {file_path} at {target_rate} Hz")
        sd.play(data, samplerate=target_rate)
        sd.wait()  # Wait until playback is finished.
    except Exception as e:
        print(f"Error playing audio: {e}")

print("\n🔊 Audio classification started.")
print("Press 'r' to label as rooster, 'n' for noise, 'd' to discard, or 'q' to quit.")

for filename in sorted(os.listdir(RECORDED_DIR)):
    if filename.endswith(".wav"):
        file_path = os.path.join(RECORDED_DIR, filename)
        print(f"\n🎵 Playing: {filename}")
        play_audio(file_path)
        
        while True:
            command = input("Enter classification (r=rooster, n=noise, d=discard, q=quit): ").strip().lower()
            if command == 'r':
                shutil.move(file_path, os.path.join(ROOSTER_DIR, filename))
                print(f"✅ Moved to: {ROOSTER_DIR}")
                break
            elif command == 'n':
                shutil.move(file_path, os.path.join(NOISE_DIR, filename))
                print(f"✅ Moved to: {NOISE_DIR}")
                break
            elif command == 'd':
                os.remove(file_path)
                print("✅ Discarded sample")
                break
            elif command == 'q':
                print("🚪 Exiting classification.")
                exit()
            else:
                print("❌ Invalid input. Use 'r' for rooster, 'n' for noise, 'd' to discard, or 'q' to quit.")

print("🎉 All recordings sorted successfully!")

