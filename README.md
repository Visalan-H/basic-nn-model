# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The following model uses linear regression to predict values of unknown data, I have used one input layer with 16 neurons, it will take one column as input.One hidden layer with 16 neurons. Both layers have relu as their activation function. Then one output layer having one neuron that gives the predicted value.
Regression models show relationships between variables, but they may not perfectly fit the data. Neural networks, though complex and computationally demanding, offer flexibility in choosing regression types and can be enhanced with hidden layers for better predictions.

## Neural Network Model

![image](https://github.com/Visalan-H/basic-nn-model/assets/152077751/98983cfb-55ce-4a48-8754-3f24432fd5f7)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Visalan H
### Register Number:212223240183
```python

212223240183


```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('datasetdl').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float','OUTPUT':'float'})
df.head()

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


X = df[['INPUT']].values
y = df[['OUTPUT']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state = 20)


Scaler = MinMaxScaler()

Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)


model=tf.keras.Sequential()

model.add(Dense(16,activation='relu',input_shape=(1,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))


model.compile(optimizer='adam',loss='mae')


model.fit(X_train1,y_train,epochs=1000)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1 = [[25.0]]


X_n1_1 = Scaler.transform(X_n1)


model.predict(X_n1_1)

## Dataset Information

![dataset1dl](https://github.com/Visalan-H/basic-nn-model/assets/152077751/9686cb42-6365-49d7-804c-1b6950939177)


## OUTPUT

### Training Loss Vs Iteration Plot

![lossplot](https://github.com/Visalan-H/basic-nn-model/assets/152077751/2274c3a3-ee86-4ed4-88d8-f93baff0265a)


### Test Data Root Mean Squared Error

![image](https://github.com/Visalan-H/basic-nn-model/assets/152077751/2160e606-8c44-40bc-a2c1-49a23c6549b2)


### New Sample Data Prediction

![image](https://github.com/Visalan-H/basic-nn-model/assets/152077751/d03951b7-45f8-41ac-b99e-9f3b3bfbc5da)


## RESULT

A neural network regression model for the given dataset has been developed successfully.
