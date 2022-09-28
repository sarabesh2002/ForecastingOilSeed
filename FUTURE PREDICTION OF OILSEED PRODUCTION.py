import pandas as pd
import warnings
import numpy as np
from matplotlib import pyplot
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statistics
import math
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv(r'C:\Users\20nis\OneDrive\Documents\dataset.csv')
train = df[0:62]
test = df[62:len(df)]

df_d = train.diff().fillna(train)
pyplot.plot(df_d)
pyplot.show()

plot_acf(df_d)
pyplot.show()

plot_pacf(df_d)
pyplot.show()


model1 = ARIMA(df, order= (1,1,0))
model_fit1 = model1.fit()
model_fit1.summary()

start = 41
end = 75
predictions = model_fit1.predict(start = start, end = end, typ = 'levels')
print(predictions)

test.mean()
rmse=sqrt(mean_squared_error(predictions,test))
print(rmse)



