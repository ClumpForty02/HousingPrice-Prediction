import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("kc_house_data.csv")
date= "datetype"
data["datetype"]= pd.to_datetime(data["date"])
columns_to_drop= ["date","sqft_lot","floors","waterfront","condition","yr_built","yr_renovated","zipcode","long","datetype"]
data.drop(columns_to_drop, axis= 1, inplace= True)
#print(data.info())

#Use X axis for all columns except the price
X = data.drop(['price'], axis=1)
#Use Y axis for only price column
Y= data['price']
#print(Y)

#Below is the data set for TRAINING ONLY i.e. 20% of the data
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2)

#Training data:
#Joining X and Y training data:
#train_data= X_train.join(Y_train)
#print(train_data)
#print(train_data.hist(figsize=(20,10)))
#plt.show()
# plt.figure(figsize= (25,5))
# sns.heatmap(train_data.corr(), annot= True, cmap= "YlGnBu")
# plt.show()

#Linear Regression
#Create the train model
train_data_joined= X_train.join(Y_train)
reg= LinearRegression()
reg.fit(X_train,Y_train)

#Predict on test data
test_data_joined= X_test.join(Y_test)
reg.score(X_test, Y_test)

predicted_prices= reg.predict(X_test)
print(predicted_prices)




