# RoosterRecorder

## Overview

A Raspberry Pi project to classify and track dedicated sound events - using a rooster cry as an example

This is a toy project which has the following goals:
* create a Raspberry Pi based device which automatically monitors, classifies and tracks dedicated sound events based on a lightweight AI approach
* on a Raspberry Pi how to
  * enable audio input and (automated recording)
  * install a lightweight AI infrastructure
  * train and use AI
  * store relevant results in a database
  * perform appealing ways to output the results
* on a metalevel, play with GPT4 etc. to see whether programming can be accelerated

## HW selection and OS installation
### HW
I decided to go for a Raspberry 5 with 8GB RAM without a dedicated HW accelerator for AI. I want to understand what level of "AI might" is possible with 
* a 4GHz Quad-Core 64-bit Arm Cortex-A76 CPU
* a VideoCore VII GPU
* and 8GB LPDDR4X-4267 SDRAM

In addition we need a microphone. I went with the cheapest USB microphone I could get. Which was called "Yosoo Helath Gear USB MINI microphone" and is barely more than the USB 2.0 connector plus a half-circle shaped piece of plastic. 

### SW installation
I installed a plain Raspberry Pi OS with desktop. Desktop just in case, I completely went headleass. 
I will not decribe this in detail, I just used the official Raspberry Pi Imager (https://www.raspberrypi.com/software/). 
Hint: before flashing the SD card use the possibility of OS customizaiton in the Imager to set SSH login, Wifi password, time zone etc. Then just insert the flashed SD card, plug in the power and let the OS do the magic. 
After a while you can login via SSH and the login data you have given before.

Then the usual hygiene steps: 
```php
sudo apt update
sudo apt full-upgrade
```
Wen need Python 3, which should be preinstalled, but just to be sure
```php
sudo apt install python3 python3-pip -y
```
### First steps with audio
First I create my playground directory. 
```php
mkdir programming
cd programming/
 ```
#### Install Audio Recording Tools

    Install ALSA Utilities (for handling audio devices):
```php
sudo apt install alsa-utils -y
```
Plug in the microphone in one of the USB ports. Now test by listing the available audio devices:
```php
arecord -l
```
In my case I got the folliwing response
```php
**** List of CAPTURE Hardware Devices ****
card 2: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```
To check if the microphone is working, record a short test audio file:
```php
arecord -D plughw:2,0 -f cd -t wav -d 5 test.wav
```
To explain the magic options: 
* -D plughw:2,0 → Specifies card 2, device 0 as the recording source.
* -f cd → Records in 16-bit, 44.1 kHz stereo format.
* -t wav → Saves as a WAV file.
* -d 5 → Records for 5 seconds.
* test.wav → Output filename.
Test the microphone:
```php
arecord -D plughw:1,0 -f cd test.wav
aplay test.wav
```
When there was just silence I realized that there the RaspPi does not have a buid-in speaker :-)
Fortunately, I had an old Jabra USB speaker in my dusty electronics leftover drawer. I plugged it in and without any further setting ajustment the "aplay" gave me sound. Yeah!

If you want the USB microphone to be the default input device, create or edit the ~/.asoundrc file:

```php
nano ~/.asoundrc
```
Add the following:

```php
pcm.!default {
    type hw
    card 2
    device 0
}

ctl.!default {
    type hw
    card 2
}
```
Save and exit (CTRL + X, then Y, then Enter).

Then, restart ALSA:

```php
sudo systemctl restart alsa-restore
```
Test again with:

```php
arecord -d 5 -f cd newtest.wav && aplay newtest.wav
```

So (hopefully) we have now a working audio setting!

## Audio Capturing

The next packages need to be installed: I tried without but using a vitual environment was finally the only way to get everything running.

### new virtual environment
Step 1: Create and Activate the Virtual Environment

    Navigate to your project directory (or home directory):

cd ~/  # Or your preferred working directory

Create a virtual environment (if not already created):

python3 -m venv myenv

Activate the virtual environment:

    source myenv/bin/activate

Your terminal should now show (myenv) at the beginning of the prompt, indicating that the virtual environment is active.
Step 2: Install sounddevice and Dependencies

    Ensure pip is up to date inside the virtual environment:

pip install --upgrade pip

Install sounddevice and necessary dependencies:

    pip install sounddevice numpy scipy

Step 3: Verify Installation

    Check if sounddevice is available:

python -c 'import sounddevice; print("Sounddevice is installed and working!")'

## Recording triggered audio

Now our first Python script. 
Copy and paste the Python code below into a file:

```php
nano trigger_audio_capture.py
```
* Press CTRL + X.
* Press Y to confirm saving.
* Press Enter to save and exit.

Here is the script
```python
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time as time_module  # Renamed to avoid conflict with datetime
import os
from datetime import datetime

# Configuration
SAMPLE_RATE = 16000               # Sampling rate in Hz
DURATION_PRE_TRIGGER = 0.5        # Seconds before trigger to keep (ring buffer)
DURATION_POST_TRIGGER = 3.5       # Seconds after trigger to record
THRESHOLD = 0.1                   # Adjust based on microphone sensitivity
COOLDOWN_PERIOD = 2               # Seconds to wait after recording completes
OUTPUT_DIR = "recorded_events"    # Folder to store audio files

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of samples for the pre-trigger portion
pre_trigger_samples = int(SAMPLE_RATE * DURATION_PRE_TRIGGER)
# Initialize ring buffer with zeros.
ring_buffer = np.zeros(pre_trigger_samples, dtype='float32')

# Use next_allowed_trigger_time to enforce a full cooldown period
next_allowed_trigger_time = 0

def apply_fade_in(audio, fade_duration, sample_rate):
    """
    Apply a fade-in effect to the beginning of the audio.
    This smooths out any abrupt transitions.
    """
    fade_samples = int(fade_duration * sample_rate)
    fade_samples = min(fade_samples, len(audio))
    fade_curve = np.linspace(0, 1, fade_samples)
    audio[:fade_samples] *= fade_curve
    return audio

def apply_fade_out(audio, fade_duration, sample_rate):
    """
    Apply a fade-out effect to the end of the audio.
    """
    fade_samples = int(fade_duration * sample_rate)
    fade_samples = min(fade_samples, len(audio))
    fade_curve = np.linspace(1, 0, fade_samples)
    audio[-fade_samples:] *= fade_curve
    return audio

def audio_callback(indata, frames, time_info, status):
    """
    Process incoming audio. When a loud sound is detected and the cooldown
    period has passed, record an event consisting of 0.5 sec pre-trigger and
    3.5 sec post-trigger audio.
    """
    global ring_buffer, next_allowed_trigger_time

    if status:
        print(f"Audio status error: {status}")

    # Get the current audio chunk (first channel)
    audio_chunk = indata[:, 0]
    current_time = time_module.time()

    # Only trigger if the threshold is exceeded and we are past the cooldown
    if np.max(np.abs(audio_chunk)) > THRESHOLD and current_time >= next_allowed_trigger_time:
        # Identify the first sample index that exceeds the threshold.
        trigger_indices = np.where(np.abs(audio_chunk) > THRESHOLD)[0]
        trigger_index = trigger_indices[0] if trigger_indices.size > 0 else 0

        print("Trigger detected! Recording event...")

        # --- Prepare the Pre-Trigger Audio ---
        # The ring buffer holds the previous 0.5 sec of audio.
        # Also include any audio from the current chunk before the trigger.
        pre_trigger_candidate = np.concatenate((ring_buffer, audio_chunk[:trigger_index]))
        pre_trigger_audio = pre_trigger_candidate[-pre_trigger_samples:]
        # Apply a short fade-out to smooth the end of the pre-trigger audio.
        pre_trigger_audio = apply_fade_out(pre_trigger_audio, fade_duration=0.02, sample_rate=SAMPLE_RATE)

        # --- Prepare the Post-Trigger Audio ---
        # Use the part of the current chunk starting from the trigger.
        post_trigger_initial = audio_chunk[trigger_index:]
        initial_length = len(post_trigger_initial)
        total_post_trigger_samples = int(SAMPLE_RATE * DURATION_POST_TRIGGER)
        remaining_samples = total_post_trigger_samples - initial_length

        if remaining_samples > 0:
            # Record additional audio to complete the post-trigger segment.
            post_trigger_remaining = sd.rec(remaining_samples,
                                            samplerate=SAMPLE_RATE,
                                            channels=1,
                                            dtype='float32')
            sd.wait()  # Wait for recording to finish.
            post_trigger_remaining = post_trigger_remaining.flatten()
        else:
            post_trigger_remaining = np.array([], dtype='float32')

        # Concatenate the immediate post-trigger audio with the extra recorded part.
        post_trigger_audio = np.concatenate((post_trigger_initial, post_trigger_remaining))
        # Apply a short fade-in to smooth the beginning of the post-trigger audio.
        post_trigger_audio = apply_fade_in(post_trigger_audio, fade_duration=0.02, sample_rate=SAMPLE_RATE)

        # --- Combine and Save ---
        # Final recording is 0.5 sec pre-trigger plus 3.5 sec post-trigger.
        combined_audio = np.concatenate((pre_trigger_audio, post_trigger_audio))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"event_{timestamp}.wav")
        # Convert audio to 16-bit PCM format.
        wav.write(filename, SAMPLE_RATE, (combined_audio * 32767).astype(np.int16))
        print(f"Saved: {filename}")

        # --- Update the Ring Buffer ---
        # Use the tail end of the post-trigger audio for future pre-trigger context.
        if len(post_trigger_audio) >= pre_trigger_samples:
            ring_buffer = post_trigger_audio[-pre_trigger_samples:]
        else:
            ring_buffer = post_trigger_audio.copy()

        # --- Update the Cooldown Timer ---
        # Now that recording is complete, disallow new triggers for the cooldown period.
        next_allowed_trigger_time = time_module.time() + COOLDOWN_PERIOD

        # Exit early so that the ring buffer update below does not override our settings.
        return

    # --- Normal Operation: Update the Ring Buffer ---
    # Always keep the ring buffer containing the last 0.5 sec of audio.
    ring_buffer = np.roll(ring_buffer, -len(audio_chunk))
    ring_buffer[-len(audio_chunk):] = audio_chunk

# --- Main Program: Start Listening ---
print("Listening for sound triggers...")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    while True:
        time_module.sleep(0.1)  # Keep the script running

```
Some explanations of the script:
### Script Overview

* Continuous Audio Monitoring:
  * The script uses the sounddevice library to continuously capture audio at 16 kHz via a callback function.
* Pre-Trigger Ring Buffer:
  * A ring buffer maintains the last 0.5 seconds of audio. This allows the script to include audio preceding the trigger event.
* Trigger Detection:
  * The script monitors incoming audio and detects a trigger when any sample exceeds a set threshold (THRESHOLD = 0.2).
* Recording Sequence:
  * Pre-Trigger Audio:
    * When a trigger is detected, the script retrieves the last 0.5 seconds of audio from the ring buffer, ensuring that the sound immediately before the trigger is captured.
  * Post-Trigger Audio:
    * It then captures the remainder of the current audio chunk starting from the trigger and records additional audio (if needed) to complete 3.5 seconds of post-trigger audio.
  * Smoothing Transitions:
    * A short fade-out is applied to the end of the pre-trigger audio and a fade-in is applied to the beginning of the post-trigger audio to smooth any abrupt transitions.
  * File Saving:
    * The final combined 4-second audio (0.5 sec pre-trigger + 3.5 sec post-trigger) is saved as a WAV file (converted to 16-bit PCM) in the recorded_events directory with a timestamped filename.
* Ring Buffer Maintenance:
  * After each trigger event, the ring buffer is updated with the tail end of the post-trigger recording to maintain accurate pre-trigger context for future events.

### Handling the Cooldown Period
* Initial Problem:
  * Originally, the cooldown period was based on the trigger time, which meant that subsequent recordings could be inadvertently triggered too soon—sometimes even when the sound level was below the threshold—because the cooldown wasn't tied to the completion of the full recording.
* Implemented Solution:
  * A new variable, next_allowed_trigger_time, is introduced.
  * Cooldown Enforcement:
    * When a trigger is processed, the script records the post-trigger audio and only after saving the file does it update next_allowed_trigger_time to the current time plus the cooldown period (2 seconds).
  * Effect:
     * This approach ensures that no new trigger will be accepted until at least 2 seconds after the 3.5-second post-trigger recording is complete, effectively spacing out recording events by approximately 5.5 seconds (trigger + 3.5 sec recording + 2 sec cooldown).
  * Outcome:
    * The updated cooldown mechanism reliably prevents overlapping or spurious triggers, ensuring that each recording is an isolated event.

By the way, this explanation was created by ChatGPT 03-mini-high.

Create the directory for the sound files
```php
mkdir recorded_events
```

And now, keep your fingers crossed, let's run the python script.
Make sure that your virtual environment ist active. The console should start with "(myenv)...".
```php 
python trigger_audio_capture.py
```

If everything goes as planned, for every sound event you get a set of time stamped .wav file in the directory recorded_event. If not then here are some fine-tuning suggestions
* Too sensitive? Increase THRESHOLD (e.g., 0.15 or 0.2).
* Not sensitive enough? Decrease THRESHOLD (e.g., 0.05).
* Too many recordings? Increase COOLDOWN_PERIOD (e.g., 3 or 4 seconds).
* Misses sounds? Increase DURATION_PRE_TRIGGER (e.g., 0.7 sec).If nothing happens then you need to decrease the THRESHOLD value. And if you get too many events, increase the THRESHOLD value. 

We have created a trigger-based sound event recorder. 

So far so good. 

## Entering the AI world

Now it is getting even more exciting. We will now create an AI-based rooster sound classification.

First, the required libiraries. Again, ensure that the virtual environment ist still active.

```php
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
```
* This downloads the latest compatible TensorFlow Lite runtime for Raspberry Pi.
* The --extra-index-url ensures that the correct Raspberry Pi-specific package is retrieved.

Now install the remaining AI packages
```php
pip install numpy librosa soundfile
```
Verify the installation
```php
python -c 'import numpy, librosa, tflite_runtime.interpreter; print("AI libraries installed and working!")'
```
If no errors appear, the installation is successful! 🚀

### Data preparation

No (classical) AI without training data. So let's create samples to be classified. 
First, put the Raspberry Pi with the microphone at the "crime scene", i.e., close to the area where the sound events happen. Then trigger our recording script. You may have to fine tune again the trigger conditions as described before. 
In my case I am lucky as I can put the Raspberry Pi in my garden but still within reach of my Wifi network. So I can make the adjustments out of my cosy office :-)

Let it run such than you get ideally more than 50 recordings eache, e.g., rooster cries vs. car sounds etc. These recordings are then stored in our diretory 'recorded_event'.

Now we have to classify them manually. Means: using a helping python script.

But first we have to identify the correct audio output device.
Execute 
```python
python -c "import sounddevice as sd; print(sd.query_devices())"
```
You will get an output like

```php
0 USB PnP Sound Device: Audio (hw:2,0), ALSA (1 in, 0 out)
  1 Jabra SPEAK 510 USB: Audio (hw:3,0), ALSA (1 in, 2 out)
  2 pulse, ALSA (32 in, 32 out)
* 3 default, ALSA (32 in, 32 out)
```

In my case Device 1 is the right one.

Create following python script, named 'sort_recordings.py'.
```python
import os
import shutil
import sounddevice as sd
import soundfile as sf

# Define directories
RECORDED_DIR = "recorded_events"
OUTPUT_DIR = "dataset"
ROOSTER_DIR = os.path.join(OUTPUT_DIR, "rooster")
NOISE_DIR = os.path.join(OUTPUT_DIR, "noise")

# Ensure output directories exist
os.makedirs(ROOSTER_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)

# Get a list of all .wav files in recorded_events
wav_files = sorted([f for f in os.listdir(RECORDED_DIR) if f.endswith(".wav")])

def play_audio(file_path):
    """Plays a given .wav file."""
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished

print("Audio classification started.")
print("Press 'r' to label as rooster, 'n' for noise, or 'q' to quit.")

for filename in wav_files:
    file_path = os.path.join(RECORDED_DIR, filename)
    
    print(f"\nPlaying: {filename}")
    play_audio(file_path)

    while True:
        command = input("Enter classification (r=rooster, n=noise, q=quit): ").strip().lower()
        
        if command == 'r':
            shutil.move(file_path, os.path.join(ROOSTER_DIR, filename))
            print(f"Moved to: {ROOSTER_DIR}")
            break
        elif command == 'n':
            shutil.move(file_path, os.path.join(NOISE_DIR, filename))
            print(f"Moved to: {NOISE_DIR}")
            break
        elif command == 'q':
            print("Exiting classification.")
            exit()
        else:
            print("Invalid input. Use 'r' for rooster, 'n' for noise, or 'q' to quit.")

print("All recordings sorted! 🎉")
```
Before running the script, create two directories:
```php
mkdir rooster
mkdir noise
```
HINT: This script did not work for me as my audio device (Jabra SPEAK 510 USB) forced a 48kHz playback, so every .wav file was played at 4x speed. Therefore, I had to add an upsampling step in the script. 

Here is my alternative, which did it for me. So you may have to tinker a bit with the script, depending on your audio settings

```python
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
```
1. Run the  python script:
```python
python sort_recordings.py
```
2. Listen to each recording:

* The script will play each .wav file in recorded_events/.

3. Manually classify:
* Press r → Moves the file to dataset/rooster/.
* Press n → Moves the file to dataset/noise/.
* Press q → Quits the script. Ideally, you do not need to press q as the script will end automatically once all .wav files are classified

Repeat until all files are classified.

Ideally, we have now two directories filled with positive and negative audio samples. So we can start training. 

Cheat: if you want to use pre-recorded sounds look at 
* https://xeno-canto.org/explore?query=Gallus%20gallus%20domesticus
* https://freesound.org/people/arundasstp/sounds/404133/
* https://research.google.com/audioset/dataset/chicken_rooster.html (videos, audio to be extracted)

You may either download the sounds or play them in front of your Raspberry Pi microphone...

## Training the AI
### Install necessary libraries
To train a sound classifier, install the required Python libraries. Again ensure that your virtual enviroment is active. This may take a while, we are talking about a >200MB download.

```php
pip install numpy librosa tensorflow scikit-learn matplotlib
```
### Create a feature extraction script
We’ll use Mel-Frequency Cepstral Coefficients (MFCCs), which are effective features for sound classification.
Save this as extract_features.py:
```python
import os
import numpy as np
import librosa
import pickle

# Adjusted paths to directories
ROOSTER_DIR = "rooster"
NOISE_DIR = "noise"
FEATURES_FILE = "features.pkl"

# Audio processing settings
SAMPLE_RATE = 16000  # Ensure consistent sampling rate
N_MFCC = 13          # Number of MFCC features

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return np.mean(mfccs, axis=1)  # Take the mean across the time axis
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Prepare dataset
X = []  # Features
y = []  # Labels: 1 = rooster, 0 = noise

# Process rooster files
for subdir, _, files in os.walk(ROOSTER_DIR):
    for filename in files:
        if filename.endswith(".wav"):
            file_path = os.path.join(subdir, filename)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(1)  # Label as rooster

# Process noise files
for subdir, _, files in os.walk(NOISE_DIR):
    for filename in files:
        if filename.endswith(".wav"):
            file_path = os.path.join(subdir, filename)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(0)  # Label as noise

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save extracted features
with open(FEATURES_FILE, "wb") as f:
    pickle.dump((X, y), f)

print(f"✅ Features saved to {FEATURES_FILE}.")
```

### Train the Classifier

Create a training script as train_classifier.py

```python
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load features
with open("features.pkl", "rb") as f:
    X, y = pickle.load(f)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Save the model
model.save("rooster_classifier.h5")
print("✅ Model saved as `rooster_classifier.h5`.")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Model Loss")

plt.show()
```

### Predicting values

Create a script to test your trained model on new audio files.
Save this as predict.py
```python
import tensorflow as tf
import librosa
import numpy as np
import sys

MODEL_FILE = "rooster_classifier.h5"
SAMPLE_RATE = 16000
N_MFCC = 13

# Load the trained model
model = tf.keras.models.load_model(MODEL_FILE)

def predict_audio(file_path):
    """Predict if the audio contains a rooster or noise."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features = np.mean(mfccs, axis=1).reshape(1, -1)  # Reshape for prediction
        prediction = model.predict(features)[0][0]
        return "Rooster 🐓" if prediction > 0.5 else "Noise 🌿"
    except Exception as e:
        return f"Error: {e}"

# Test the model on a file
if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_wav_file>")
else:
    file_path = sys.argv[1]
    result = predict_audio(file_path)
    print(f"🔊 Prediction: {result}")
```


### Test Your Classifier

Run the following commands:

1. Extract Features:
```php
python extract_features.py
```
Train the Classifier:
```php
python train_classifier.py
```
Predict New Audio:
```php
python predict.py triggers/test.wav
```
Replace test.wav with a new .wav file to classify.


## perhaps needs to be inserted above 

pip install noisereduce


# Helpful hints and further support
## Activating Samba
When I put the RasPi in my garden shed I obviously had to access it remoteley. Thus I installed Samba for an easier access.
Check if it is already running
```php
sudo systemctl status smbd
```
If not - as it was the case for me - install it.
```php
sudo apt update && sudo apt install samba -y
```
In my case there were no samba login data (check as follows):
```php
pdbedit -L
```
So I had to add it and restart the service .
```php
sudo rm -f /var/lib/samba/private/passdb.tdb
sudo smbpasswd -a <your_samba_login_password>
sudo systemctl restart smbd nmbd
```
This gives us now thw opportunity to play .wav files remotely on a Mac/PC - and to create and edit python scripts directly in Visual Studio Code, which is much more convenient than via a simple terminal editor like vi or nano. And you can play even the .wav files directly withon Visual Studio Code!
