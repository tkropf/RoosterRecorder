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
