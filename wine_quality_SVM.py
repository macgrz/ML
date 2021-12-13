### DATA USED: https://archive.ics.uci.edu/ml/datasets/Wine+Quality ###
### ML with white wine quality data set
### Basic pipeline with SVM classification and StandardScaler
### Comparisson of Accuracy (scaled and unscaled)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100], 'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the scaled training set
svm_scaled = cv.fit(X_train, y_train)

# Instantiate and fit a SVM classifier to the unscaled data
svm_unscaled = SVC().fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = svm_scaled.predict(X_test)

# Compute and print metrics of scaled SVM
print("Accuracy (scaled): {}".format(svm_scaled.score(X_test, y_test)))
print("Tuned Model Parameters (scaled): {}".format(svm_scaled.best_params_))
# print(classification_report(y_test, y_pred))

# Compute and print metrics of unscaled SVM
print('Accuracy (unscaled): {}'.format(svm_unscaled.score(X_test, y_test)))
