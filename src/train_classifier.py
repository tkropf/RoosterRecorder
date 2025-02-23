import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the extracted features from CSV
df = pd.read_csv("features.csv")

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode the labels (e.g., 'rooster' and 'noise') to numeric values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the trained model and the label encoder for later use on the Raspberry Pi
model_filename = "rooster_classifier.pkl"
with open(model_filename, "wb") as f:
    pickle.dump((clf, le), f)

print(f"Model saved as {model_filename}")
