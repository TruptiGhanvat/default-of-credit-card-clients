import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the cleaned dataset
df = pd.read_csv("C:///sers///TRUPTI VAMAN GHANVAT///eDriveDesktop///Scredit_default_sample.csv")

# Features and target
X = df.drop('default_payment_next_month', axis=1)
y = df['default_payment_next_month']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
