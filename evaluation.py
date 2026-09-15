# Model Evaluation

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load dataset
df = pd.read_csv("your_file.csv")


# 2. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Create model
model = LogisticRegression()


# 5. Train model
model.fit(X_train, y_train)


# 6. Prediction
y_pred = model.predict(X_test)


# 7. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# 8. Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(confusion)


# 9. Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)