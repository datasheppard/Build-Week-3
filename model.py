# Importing the libraries

import pandas as pd
import pickle
import requests
import category_encoders as ce
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
import numpy as np
import json  # Importing the dataset

df = df.dropna()

X = df.drop('taxvaluedollarcnt')
y = df['taxvaluedollarcnt']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.20,
                                                  random_state=42)

target = 'taxvaluedollarcnt'

encoder = ce.OrdinalEncoder()
xtre = encoder.fit_transform(X_train)
xve = encoder.transform(X_val)

X_test = encoder.transform(X_test)

eval_set = [(xtre, y_train), (xve, y_val)]

model = XGBRegressor(n_estimators=50, n_jobs=-1)
model.fit(xtre, y_train, eval_set=eval_set,
          eval_metric='rmse', early_stopping_rounds=20)

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[4, 4, 96268]]))
