import os
import librosa
import numpy as np
import pandas as pd

# Directories containing the audio samples
ROOSTER_DIR = 'rooster'
NOISE_DIR = 'noise'
OUTPUT_CSV = 'features.csv'

# Sampling rate for loading audio (our recordings are at 16 kHz)
TARGET_SR = 16000

def extract_features(file_path, n_mfcc=13):
    """
    Extracts MFCC features from an audio file.
    
    Parameters:
      file_path (str): Path to the audio file.
      n_mfcc (int): Number of MFCC coefficients to extract.
    
    Returns:
      np.ndarray: A 1D array containing the mean and standard deviation
                  of each MFCC coefficient (length = n_mfcc*2).
    """
    try:
        # Load audio file (mono) at the target sampling rate
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        
        # Compute MFCCs; shape: (n_mfcc, frames)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Compute the mean and standard deviation for each coefficient
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Concatenate the mean and std vectors to form the feature vector
        features = np.concatenate((mfccs_mean, mfccs_std))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    return features

def process_directory(directory, label, data_list, label_list):
    """
    Processes all WAV files in a directory, extracting features and appending
    them along with the provided label.
    
    Parameters:
      directory (str): Path to the directory.
      label (str): Label to assign to all files in this directory.
      data_list (list): List to append the feature vectors.
      label_list (list): List to append the labels.
    """
    for file in os.listdir(directory):
        if file.lower().endswith('.wav'):
            file_path = os.path.join(directory, file)
            features = extract_features(file_path)
            if features is not None:
                data_list.append(features)
                label_list.append(label)

# Lists to hold feature vectors and labels
features_data = []
labels = []

# Process the two directories
process_directory(ROOSTER_DIR, 'rooster', features_data, labels)
process_directory(NOISE_DIR, 'noise', features_data, labels)

# Convert the lists to NumPy arrays for further processing
features_data = np.array(features_data)
labels = np.array(labels)

# Create column names (first half: MFCC means, second half: MFCC stds)
n_features = features_data.shape[1]
n_mfcc = n_features // 2
columns = [f"mfcc_{i+1}_mean" for i in range(n_mfcc)] + [f"mfcc_{i+1}_std" for i in range(n_mfcc)]

# Create a DataFrame and add the label column
df = pd.DataFrame(features_data, columns=columns)
df['label'] = labels

# Save the features to a CSV file
df.to_csv(OUTPUT_CSV, index=False)
print(f"Feature extraction completed. Features saved to {OUTPUT_CSV}")
