import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# import the dataframe
dataset_ready_select_features =pd.read_csv('df_mtl_features_engineering.csv')

## Split data and feature slection
from sklearn.model_selection import train_test_split

#Defining the independent variables and dependent variables
airbnb_en=dataset_ready_select_features.copy()
airbnb_en = airbnb_en.loc[(airbnb_en['price'] < 900)]
x = airbnb_en.drop(['price'], axis=1)

# use log10 for the price for a good result
y = airbnb_en['price'].values
y = np.log10(y)

#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=42)

selected_feat=['neighbourhood','property_type', 'room_type', 'accommodates', 'bedrooms', 'beds', 'bed_type',
       'availability_365', 'instant_bookable', 'host_is_superhost', 'number_of_reviews', 'review_scores_rating']

x_train=x_train[selected_feat]
x_test =x_test[selected_feat]


# LR Prediction Model
from sklearn.linear_model import LinearRegression

#Prepare a Linear Regression (LR) Model
reg=LinearRegression()
reg.fit(x_train,y_train)

# Saving model to disk
pickle.dump(reg, open('model.pkl','wb')) 
