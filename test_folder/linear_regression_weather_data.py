from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.linear_model import LinearRegression
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta


df = pd.read_csv('WeatherData.csv')
df.set_index('date')
features = list(df)


def nth_day_coulumn_data(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_prior_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

for feature in features:
    if feature != 'date':
        for N in range(1, 6):
            nth_day_coulumn_data(df, feature, N)

df = df.dropna()
df = df.apply(pd.to_numeric, errors='coerce')

print df.corr()[['meantempm']].sort_values('meantempm')

col_high_coeff = ['meantempm_prior_1',  'meantempm_prior_2','meantempm_prior_3','meantempm_prior_4',
                  'meantempm_prior_5',
                  'mintempm_prior_1',   'mintempm_prior_2', 'mintempm_prior_3' , 'mintempm_prior_4',
                  'mintempm_prior_5',
                  'meandewptm_prior_1', 'meandewptm_prior_2', 'meandewptm_prior_3' , 'meandewptm_prior_4' ,
                  'meandewptm_prior_5',
                  'maxdewptm_prior_1',  'maxdewptm_prior_2', 'maxdewptm_prior_3' , 'maxdewptm_prior_4' ,
                  'maxdewptm_prior_5',
                  'mindewptm_prior_1',  'mindewptm_prior_2', 'mindewptm_prior_3' , 'mindewptm_prior_4',
                  'mindewptm_prior_5',
                  'maxtempm_prior_1',   'maxtempm_prior_2' , 'maxtempm_prior_3','maxtempm_prior_4',
                  'maxtempm_prior_5']
new_df = df[['meantempm'] + col_high_coeff]

print new_df.columns
#space3

for col in new_df.columns:
    # create a boolean array of values representing nans
    missing_vals = pd.isnull(df[col])
    df[col][missing_vals] = 0

new_df.dropna


X = new_df[col_high_coeff]
y = new_df['meantempm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
model = LinearRegression()
# fit the build the model by fitting the regressor to the training data
model.fit(X_train, y_train)
# make a prediction set using the test set
prediction = model.predict(X_test)
# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % model.score(X_test, y_test))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

#testing with a date
date_to_evaluate = datetime.today() - timedelta(1000)
print date_to_evaluate.datetime.timestamp()
#test_date=date_to_evaluate.strftime('%d/%m/%y 0:00')
X_test_data = df.loc[df['date'] == date_to_evaluate.datetime.timestamp()]
#print model.predict(X_test_data)




