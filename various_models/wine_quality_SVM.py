### DATA USED: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
### ML with white wine quality dataset
### Basic pipeline with SVM classification and StandardScaler and 3-fold cross-validation
### Comparisson of models (untuned and tuned)

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

# Instantiate the GridSearchCV object with scaled data: svm_cv
gm_scaled_tuned = GridSearchCV(pipeline, param_grid=parameters, cv=3)

# Instantiate and fit a tuned SVM classifier to the scaled data
gm_scaled_tuned.fit(X_train, y_train)

# Instantiate and fit a untuned SVM classifier to the unscaled data
m_unscaled = SVC().fit(X_train, y_train)

# Predict the labels of the test set for both
y_pred_tuned = gm_scaled_tuned.predict(X_test)
y_pred_untuned = m_unscaled.predict(X_test)

# Compute and print metrics of scaled, tuned SVM with best parameters and classification_report
print("Accuracy (scaled, tuned): {}".format(gm_scaled_tuned.score(X_test, y_test)))
print("Tuned Model Parameters (scaled): {}".format(gm_scaled_tuned.best_params_))
print(classification_report(y_test, y_pred_tuned))

# Compute and print metrics of unscaled, untuned SVM with classification_report
print('Accuracy (unscaled, untuned): {}'.format(m_unscaled.score(X_test, y_test)))
print(classification_report(y_test, y_pred_untuned))
