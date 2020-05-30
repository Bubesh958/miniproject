import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request as urllib2
from bs4 import BeautifulSoup
import json



df = pd.read_csv('dataset_cleaned.csv')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]




y = df['Value']
#df.shape



X = df.drop(['Value'], axis=1)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)




from xgboost.sklearn import XGBRegressor

xboost = XGBRegressor(n_estimators=200)





xboost.fit(X_train, y_train)






xgb_score = xboost.score(X_test, y_test)


print(xgb_score)