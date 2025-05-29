# naive_bayes_crop_visual.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt

# Step 1: Load the dataset
print("🔄 Loading crop dataset...")
data = pd.read_csv("crop_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Step 2: Encode categorical features
print("🔠 Encoding features...")
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)
y_encoded = le.fit_transform(y)

# Step 3: Split the data
print("🧪 Splitting into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42
)

# Step 4: Train Naive Bayes model
print("📌 Training Gaussian Naive Bayes model...")
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
print("📈 Making predictions...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2f}")

# Step 6: Accuracy Measures
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 7: Visualize confusion matrix
print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Greens)
plt.title("Naive Bayes - Crop Success Prediction")
plt.show()
