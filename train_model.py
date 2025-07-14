from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, "iris_model.pkl")

print("Model trained and saved as iris_model.pkl")
