import os
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np
import argparse

# Directories:
# RECORDED_DIR should be the local copy of the "recorded_events" folder from the Raspberry Pi.
RECORDED_DIR = "recorded_events"
ROOSTER_DIR = "rooster"
NOISE_DIR = "noise"

# Ensure the output directories exist.
os.makedirs(ROOSTER_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)

def play_audio(file_path, playback_speed):
    """
    Plays a .wav file, adjusting playback speed by resampling.
    """
    try:
        data, samplerate = sf.read(file_path)
        target_rate = int(samplerate * playback_speed)  # Adjust target rate based on playback speed

        # Resample the data
        new_length = int(len(data) / playback_speed)
        resampled_data = np.interp(
            np.linspace(0, len(data), new_length, endpoint=False),
            np.arange(len(data)),
            data
        )
        
        print(f"Playing {file_path} at {target_rate} Hz (x{playback_speed} speed)")
        sd.play(resampled_data, samplerate=samplerate)  # Play at original sample rate
        sd.wait()  # Wait until playback is finished
    except Exception as e:
        print(f"Error playing audio: {e}")

def main(playback_speed):
    """
    Main function to classify audio recordings.
    """
    print("\n🔊 Audio classification started.")
    print("Press 'r' to label as rooster, 'n' for noise, 'd' to discard, or 'q' to quit.")

    for filename in sorted(os.listdir(RECORDED_DIR)):
        if filename.endswith(".wav"):
            file_path = os.path.join(RECORDED_DIR, filename)
            print(f"\n🎵 Playing: {filename}")
            play_audio(file_path, playback_speed)  # Pass the user-defined speed
            
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

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sort audio recordings with adjustable playback speed.")
    parser.add_argument("-speed", type=float, default=1.0, help="Set the playback speed (e.g., 2 for double speed, 0.5 for half speed).")
    args = parser.parse_args()

    # Call the main function with the specified playback speed
    main(playback_speed=args.speed)
