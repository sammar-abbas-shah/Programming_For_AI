from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Generate an imbalanced classification dataset
X, y = make_classification(n_samples=10000, weights=[0.95], flip_y=0, random_state=2)
print(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#Synthetic Minority Over-sampling Technique
# Apply SMOTE to the training set
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(y_train.shape)

# Train a logistic regression model on the resampled training set
model = LogisticRegression()
model1 = LogisticRegression()

model.fit(X_train,y_train)
model1.fit(X_train_resampled, y_train_resampled)

# Evaluate the performance of the trained model on the testing set
y_pred = model.predict(X_test)
y_pred_res = model1.predict(X_test)

print("STATS-WITHOUT SMOTE")
print(classification_report(y_test, y_pred))
print("STATS-WITH SMOTE")
print(classification_report(y_test, y_pred_res))

print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred_res))