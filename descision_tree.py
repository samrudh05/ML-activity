# decision_tree_visual.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
print("🔄 Loading dataset...")
data = pd.read_csv("crop_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode categorical features
print("🔠 Encoding features...")
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)
y_encoded = le.fit_transform(y)

# Split into training and testing sets
print("🧪 Splitting into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Train the Decision Tree
print("🌳 Training Decision Tree...")
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X_train, y_train)

# Predictions and Accuracy
print("📈 Making predictions...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.2f}\n")

# Show decision tree in text format
print("📄 Textual Decision Tree:\n")
print(export_text(clf, feature_names=list(X.columns)))

# Classification Report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=list(X.columns), class_names=list(le.classes_), filled=True, rounded=True)
plt.title("🌳 Decision Tree - Crop Suitability")
plt.show()

# Confusion Matrix
print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Oranges)
plt.title("Decision Tree - Confusion Matrix")
plt.show()
