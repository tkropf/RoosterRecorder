import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time as time_module
import os
from datetime import datetime, timedelta
import pickle
import sqlite3
import librosa
import argparse

# ================================
# Parse Command Line Options
# ================================
parser = argparse.ArgumentParser(description="AI Audio Classifier")
parser.add_argument("-r", "--rooster", action="store_true",
                    help="Store rooster sound events (wav files) in rooster_ai_classified")
parser.add_argument("-n", "--noise", action="store_true",
                    help="Store noise sound events (wav files) in noise_ai_classified")
args = parser.parse_args()

# ================================
# Configuration
# ================================
SAMPLE_RATE = 16000               # Sampling rate in Hz
DURATION_PRE_TRIGGER = 0.5        # Seconds before trigger to keep (ring buffer)
DURATION_POST_TRIGGER = 3.5       # Seconds after trigger to record
THRESHOLD = 0.1                   # Threshold for triggering
COOLDOWN_PERIOD = 2               # Seconds to wait after recording completes

# Directories for saving files
RECORD_DIR = "recorded_events"          # Temporary storage of raw events
ROOSTER_DIR = "rooster_ai_classified"     # Classified as rooster
NOISE_DIR = "noise_ai_classified"         # Classified as noise

# Database file for logging events
DB_FILE = "classification_events.db"

# ================================
# Setup Directories & Database
# ================================
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(ROOSTER_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)

# Initialize SQLite database (creates table if not exists) with check_same_thread=False
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        label TEXT,
        confidence REAL,
        file_path TEXT
    )
''')
conn.commit()

# ================================
# Load Pre-trained Classifier
# ================================
with open("rooster_classifier.pkl", "rb") as f:
    clf, le = pickle.load(f)

# ================================
# Initialize Audio Ring Buffer & Cooldown
# ================================
pre_trigger_samples = int(SAMPLE_RATE * DURATION_PRE_TRIGGER)
ring_buffer = np.zeros(pre_trigger_samples, dtype='float32')
next_allowed_trigger_time = 0

# ================================
# Helper Functions
# ================================
def apply_fade_in(audio, fade_duration, sample_rate):
    """Apply a fade-in effect to the beginning of the audio."""
    fade_samples = int(fade_duration * sample_rate)
    fade_samples = min(fade_samples, len(audio))
    fade_curve = np.linspace(0, 1, fade_samples)
    audio[:fade_samples] *= fade_curve
    return audio

def apply_fade_out(audio, fade_duration, sample_rate):
    """Apply a fade-out effect to the end of the audio."""
    fade_samples = int(fade_duration * sample_rate)
    fade_samples = min(fade_samples, len(audio))
    fade_curve = np.linspace(1, 0, fade_samples)
    audio[-fade_samples:] *= fade_curve
    return audio

def extract_features_from_audio_array(y, sr, n_mfcc=13):
    """
    Extracts MFCC features from an audio array.
    Returns a feature vector (concatenated means and std deviations).
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    features = np.concatenate((mfccs_mean, mfccs_std))
    return features

def classify_audio(file_path):
    """
    Loads an audio file, extracts features, and classifies it.
    Returns the predicted label and the confidence score.
    """
    # Load audio file (mono) with the target sample rate
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    features = extract_features_from_audio_array(y, sr)
    features = features.reshape(1, -1)
    prediction = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]
    confidence = np.max(proba)  # highest probability as confidence
    label = le.inverse_transform([prediction])[0]
    return label, confidence

# ================================
# Audio Callback Function
# ================================
def audio_callback(indata, frames, time_info, status):
    global ring_buffer, next_allowed_trigger_time

    if status:
        print(f"Audio status error: {status}")

    # Get current audio chunk from the first channel
    audio_chunk = indata[:, 0]
    current_time = time_module.time()

    # Update ring buffer continuously with the latest audio chunk
    ring_buffer = np.roll(ring_buffer, -len(audio_chunk))
    ring_buffer[-len(audio_chunk):] = audio_chunk

    # Check if any sample in the chunk exceeds the threshold and cooldown has passed
    if np.max(np.abs(audio_chunk)) > THRESHOLD and current_time >= next_allowed_trigger_time:
        # Determine the trigger sample index
        trigger_indices = np.where(np.abs(audio_chunk) > THRESHOLD)[0]
        trigger_index = trigger_indices[0] if trigger_indices.size > 0 else 0
        print("Trigger detected! Recording event...")

        # --- Prepare Pre-trigger Audio ---
        pre_trigger_candidate = np.concatenate((ring_buffer, audio_chunk[:trigger_index]))
        pre_trigger_audio = pre_trigger_candidate[-pre_trigger_samples:]
        pre_trigger_audio = apply_fade_out(pre_trigger_audio, fade_duration=0.02, sample_rate=SAMPLE_RATE)

        # --- Record Post-trigger Audio ---
        post_trigger_initial = audio_chunk[trigger_index:]
        initial_length = len(post_trigger_initial)
        total_post_trigger_samples = int(SAMPLE_RATE * DURATION_POST_TRIGGER)
        remaining_samples = total_post_trigger_samples - initial_length

        if remaining_samples > 0:
            post_trigger_remaining = sd.rec(remaining_samples,
                                            samplerate=SAMPLE_RATE,
                                            channels=1,
                                            dtype='float32')
            sd.wait()  # Wait until recording is complete
            post_trigger_remaining = post_trigger_remaining.flatten()
        else:
            post_trigger_remaining = np.array([], dtype='float32')

        post_trigger_audio = np.concatenate((post_trigger_initial, post_trigger_remaining))
        post_trigger_audio = apply_fade_in(post_trigger_audio, fade_duration=0.02, sample_rate=SAMPLE_RATE)

        # --- Combine Pre- and Post-trigger Audio ---
        combined_audio = np.concatenate((pre_trigger_audio, post_trigger_audio))

        # Save the event as a WAV file in the temporary record directory
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"event_{timestamp_str}.wav"
        file_path = os.path.join(RECORD_DIR, file_name)
        wav.write(file_path, SAMPLE_RATE, (combined_audio * 32767).astype(np.int16))
        print(f"Saved raw event: {file_path}")

        # --- Classify the Recorded Audio ---
        label, confidence = classify_audio(file_path)
        print(f"Classification result: {label} with confidence {confidence:.2f}")

        # --- Store or Delete the File Based on Command-Line Options ---
        stored_path = ""
        if label.lower() == "rooster":
            if args.rooster:
                target_dir = ROOSTER_DIR
                target_path = os.path.join(target_dir, file_name)
                os.rename(file_path, target_path)
                stored_path = target_path
                print(f"Moved rooster file to: {target_path}")
            else:
                os.remove(file_path)
                print("Rooster storage disabled. File not stored.")
        else:
            if args.noise:
                target_dir = NOISE_DIR
                target_path = os.path.join(target_dir, file_name)
                os.rename(file_path, target_path)
                stored_path = target_path
                print(f"Moved noise file to: {target_path}")
            else:
                os.remove(file_path)
                print("Noise storage disabled. File not stored.")

        # --- Log the Event in the Database ---
        event_timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO events (timestamp, label, confidence, file_path) VALUES (?, ?, ?, ?)", 
                       (event_timestamp, label, confidence, stored_path))
        conn.commit()

        # Update the ring buffer with the tail end of the post-trigger audio
        if len(post_trigger_audio) >= pre_trigger_samples:
            ring_buffer = post_trigger_audio[-pre_trigger_samples:]
        else:
            ring_buffer = post_trigger_audio.copy()

        # Set cooldown timer to avoid immediate re-triggering
        next_allowed_trigger_time = time_module.time() + COOLDOWN_PERIOD
        return

# ================================
# Main Loop: Start Listening
# ================================
print("Listening for sound triggers...")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    while True:
        time_module.sleep(0.1)
