### ML with london_bike_sharing.csv dataset
### Basic LinearRegression with visualization


    # Importing CSV as DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

PATH = #INSERT PATH TO CSV FILE HERE

data_raw = pd.read_csv(PATH, sep = ",")

    # Checking data

#print(data_raw.head())
#print(data_raw.isna().any())

data = data_raw

    # Linear regression of all dataset

x = data.drop(['timestamp', 'cnt'], axis=1)
y = data['cnt']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
model = LinearRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)

MSE = round(metrics.mean_squared_error(y_test, predict), 2)
R2 = round(metrics.r2_score(y_test, predict), 4)
coef = model.coef_

print("\n\n\nLinear regression results:")
print(f"Mean squared error (MSE): {MSE}")
print(f"Coefficient of determination (R2): {R2}")
print(f"Coefficients: {coef}")
plt.scatter(predict, y_test, color='b')
plt.show()

data[['year', 'hour']] = data['timestamp'].str.split(' ', expand=True)
data['year'] = data['year'].str[:-6]

    # Relevance levels

column_names = list(data.keys())

X = sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
results.params
p = results.pvalues
p = p.round(5)
print('\nRelevance levels:')
print(p[p<=0.05])

    # Corr matrix

corrMatrix = data.corr()
fig, ax = plt.subplots(figsize=(10,8))
ax = sns.heatmap(corrMatrix, linewidths=0.5, annot=True)
plt.xticks(rotation=45) 
plt.show()

    # Linear Regression as function

def getdf(year):
    data[data['year'] == str(year)]

def linearreg(dataset, year):
    
    if str(year) not in data['year'].unique():
        raise ValueError (f'{str(year)} not in the dataset!')

    else:
        getdf(year)
        x = dataset.drop(['timestamp', 'cnt', 'hour', 'year'], axis=1)
        y = dataset['cnt']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
        model = LinearRegression()
        model.fit(x_train, y_train)
        predict = model.predict(x_test)

        MSE = round(metrics.mean_squared_error(y_test, predict), 2)
        R2 = round(metrics.r2_score(y_test, predict), 4)
        coef = model.coef_

        print(f"\n{str(year)} linear regression results:")
        print(f"Mean squared error (MSE): {MSE}")
        print(f"Coefficient of determination (R2): {R2}")
        #print(f"Coefficients: {coef}")

linearreg(data, 2015)
linearreg(data, 2016)
linearreg(data, 2017)

print('\n')
