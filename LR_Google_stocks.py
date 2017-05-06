import pandas as pa
import math, datetime
import numpy as np
import csv
from numpy import *
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


#making dataframe of stock prices
df = pa.read_csv('google_stocks.csv', sep = '\t')

#formatting data frame according to the need
del df['Previous Day Price']
df.columns = ['Date', 'High', 'Low','Last', 'Volatility']
#making volatility a feature
df['Volatility'] = df['High'] - df['Low']

df.fillna(-9999, inplace=True)


#Here we are using low of last day and volatility as label to predict the high of next Day
#label will be the high of next 10 Day

percent_pred_used = int(math.ceil(0.1*len(df)))
print percent_pred_used
df['Fut_Vol'] = df['Volatility'].shift(-percent_pred_used)
print df['Fut_Vol']

#droping those rows which do not have any value of Fut_Vol due to the shift
df.dropna(inplace=True)


#features is everything except Low and date and Last i.e High and Volatility
xf = df.drop(['Low', 'Date', 'Last'],1)

X = (xf.values).astype(int)
X = X[:-percent_pred_used]
X_lately = X[-percent_pred_used:]
#label is Fut_Val
y = np.array(df['Fut_Vol'])
y = y[:-percent_pred_used]

# 20 percent of the data is used as testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = LinearRegression(n_jobs = -1) #n_jobs is for threading
#train model
clf.fit(X_train, y_train)
#test model
accuracy = clf.score(X_test, y_test)


#predicting

forecast = clf.predict(X_lately)

print(forecast, accuracy)
