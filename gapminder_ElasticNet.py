### ML with Gapminder data set
### Basic pipeline with ElasticNet, StandardScaler, Imputation and 3-fold cross-validation
### Comparisson of models (untuned and tuned)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv_scaled_tuned
gm_cv_scaled_tuned = GridSearchCV(pipeline, param_grid=parameters, cv=3)

# Instantiate and fit a tuned ElasticNet to the scaled data
gm_cv_scaled_tuned.fit(X_train, y_train)

# Instantiate and fit a ElasticNet to the untuned model and unscaled data with no CV
m_untuned = ElasticNet().fit(X_train, y_train)

# Compute and print the metrics
r2_scaled_tuned = gm_cv_scaled_tuned.score(X_test, y_test)
r2_unscaled_untuned = m_untuned.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv_scaled_tuned.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2_scaled_tuned))
print("Untuned ElasticNet R squared: {}".format(r2_unscaled_untuned))
