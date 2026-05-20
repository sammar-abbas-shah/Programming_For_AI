from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence

# Perform k-fold cross-validation (let's use k=5 for this example)
k = 5
scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')

# Print the accuracy scores for each fold
print("Accuracy Scores for each fold:", scores)

# Calculate the mean and standard deviation of the accuracy scores
mean_accuracy = scores.mean()
std_accuracy = scores.std()

print("\nMean Accuracy:", mean_accuracy)
print("Standard Deviation of Accuracy:", std_accuracy)
