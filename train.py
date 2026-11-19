import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import normalize

# Load data
df = pd.read_csv("Data/landmarks.csv", header=None)
print("Samples per class:\n", df[0].value_counts().sort_index())

y = df.iloc[:, 0].values
X = np.array([normalize(row) for row in df.iloc[:, 1:].values])

# Split 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Train
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("\nTest accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(y_test, pred))

# Save
joblib.dump(model, "Models/asl_model.pkl")
print("Model saved to Models/asl_model.pkl")