# knn_crop_visual.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt

# Step 1: Load Data
print("🔄 Loading dataset...")
data = pd.read_csv("crop_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Step 2: Encode categorical data
print("🔠 Encoding categorical values...")
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)
y_encoded = le.fit_transform(y)

# Step 3: Split into training and test sets
print("🧪 Splitting data (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42
)

# Step 4: Apply KNN
print("📌 Applying KNN (k=3)...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Predictions and accuracy
predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Accuracy: {accuracy:.2f}")

# Step 6: Accuracy Measures
print("\n📈 Classification Report:")
print(classification_report(y_test, predictions, target_names=le.classes_))

# Step 7: Confusion Matrix Visualization
print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("KNN - Crop Success Prediction")
plt.show()
