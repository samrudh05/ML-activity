# 🌾 Agriculture Crop Suitability Classification Using Machine Learning

This project applies supervised machine learning algorithms to classify whether environmental conditions (like soil type, rainfall, temperature, and season) are suitable for growing crops. The goal is to help farmers and agricultural systems make informed decisions based on historical and environmental data.

---

## 📁 Dataset Description

The dataset contains **20,000+ records** and includes the following features:

| Feature      | Description                             |
|--------------|-----------------------------------------|
| Soil         | Type of soil (Loamy, Sandy, Clay, etc.) |
| Rainfall     | Rainfall level (Low, Medium, High)      |
| Temperature  | Temperature condition (Cool, Warm, Hot) |
| Season       | Season of the year                      |
| CropSuccess  | Target label: `Yes` or `No`             |

All features are **categorical** and require encoding before training.

---

## 🧠 Algorithms Used

- ✅ K-Nearest Neighbour (KNN)
- ✅ Decision Tree (ID3)
- ✅ Support Vector Machine (SVM)
- ✅ Naive Bayes Classifier
- ✅ Logistic Regression

Each model is trained, tested, and evaluated using accuracy, confusion matrix, and classification reports.

---

## 🧪 Project Structure

