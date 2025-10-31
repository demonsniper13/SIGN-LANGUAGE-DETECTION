import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Check if dataset exists
if not os.path.exists('./data.pickle'):
    print("❌ Error: data.pickle not found. Run create_dataset.py first.")
    exit()

# Load dataset
print("📂 Loading dataset...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"✅ Dataset loaded: {len(data)} samples")
print(f"   Feature dimensions: {data.shape}")
print(f"   Unique classes: {len(np.unique(labels))}")

# Validate data consistency
feature_lengths = [len(sample) for sample in data_dict['data']]
if len(set(feature_lengths)) > 1:
    print(f"⚠️ Warning: Inconsistent feature lengths detected: {set(feature_lengths)}")
    print("   Filtering to most common length...")
    most_common_length = max(set(feature_lengths), key=feature_lengths.count)
    filtered_data = []
    filtered_labels = []
    for i, sample in enumerate(data_dict['data']):
        if len(sample) == most_common_length:
            filtered_data.append(sample)
            filtered_labels.append(data_dict['labels'][i])
    data = np.asarray(filtered_data)
    labels = np.asarray(filtered_labels)
    print(f"   ✅ Filtered to {len(data)} samples with {most_common_length} features")

# Split dataset
print("\n🔀 Splitting dataset (80% train, 20% test)...")
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

print(f"   Training samples: {len(x_train)}")
print(f"   Testing samples: {len(x_test)}")

# Train model
print("\n🤖 Training Random Forest classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)
print("   ✅ Training complete!")

# Evaluate model
print("\n📊 Evaluating model...")
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"\n🎯 Accuracy: {score * 100:.2f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_predict))

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✅ Model saved to model.p")
print("\n🎉 Training complete! Run inference_classifier.py to test the model.")