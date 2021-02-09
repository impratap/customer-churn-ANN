# -*- coding: utf-8 -*-
"""
Creating an ANN model to predict customer churn by using tensorflow library. 
1. First will import all library.
2. load data
3. dividing data into dependent and independent variables
4. data cleaning , Encode target labels, feature scaling
5. Spliting data through train_test_split
6. Creating model.
7. Compiling model.
8. Predicting model
9. After prediction getting confusion metrix and accuracy score.
"""

# Importing all necessary Libraries 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Loading data set
dataset=pd.read_csv('Customer_Churn_Modelling.csv')

dataset.head()

# Differentiating data between dependent variable and independent variables

X=dataset.iloc[:,3:13]  # Independent variable

y=dataset['Exited'] # dependent variable

X.head()

# Encode target labels 
#LabelEncoder can be used to normalize labels. It can also be used to transform non-numerical labels to numerical labels

from sklearn.preprocessing import LabelEncoder
label1=LabelEncoder()
X['Geography']=label1.fit_transform(X['Geography'])

label2=LabelEncoder()
X['Gender']=label2.fit_transform(X['Gender'])

X.head()

# Getting dummy
X=pd.get_dummies(X,drop_first=True,columns=['Geography'])

X.head()

# feature scaling
from sklearn.preprocessing import StandardScaler

# dividing data into  X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0,stratify=y)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# creating model
model=Sequential()

# First layer

model.add(Dense(6, kernel_initializer='he_uniform', activation='relu',input_dim=X.shape[1]))

# Second layer
model.add(Dense(6, kernel_initializer='he_uniform', activation='relu'))

# third layer (Output layer)
model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

# getting model summary
model.summary()

# Compiling model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting model
compute=model.fit(X_train,y_train,validation_split=0.33, batch_size=10,epochs=20,verbose=1)

print(compute.history.keys())

# Graphical presentation of accuracy and validation accuracy

plt.plot(compute.history['accuracy'])
plt.plot(compute.history['val_accuracy'])
plt.title('Model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Graphical presentation of loss and validation loss

plt.plot(compute.history['loss'])
plt.plot(compute.history['val_loss'])
plt.title('Model_accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# getting prediction
y_pred=model.predict_classes(X_test)
y_pred

# evaluating model
model.evaluate(X_test,y_test)

# Getting confuion metrix by using  of seaborn
from sklearn.metrics import confusion_matrix

import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# printing accuracy
from sklearn.metrics  import accuracy_score
accuracy_score(y_test,y_pred)