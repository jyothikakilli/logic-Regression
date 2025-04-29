import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Creating a synthetic dataset
np.random.seed(42)
data_size = 200

X1 = np.random.rand(data_size) * 10  # Random feature 1
X2 = np.random.rand(data_size) * 5   # Random feature 2
Y = (X1 + X2 > 7).astype(int)  # Generating labels: Binary classification (0 or 1)

df = pd.DataFrame({"Feature1": X1, "Feature2": X2, "Target": Y})
print(df.head())
X = df[["Feature1", "Feature2"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
plt.scatter(X_test["Feature1"], X_test["Feature2"], c=y_pred, cmap='coolwarm', edgecolors='black')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Classification")
plt.show()