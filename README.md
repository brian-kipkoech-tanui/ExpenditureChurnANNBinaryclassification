# ANN Classification

### Importing the libraries


```python
import numpy as np
import pandas as pd
import tensorflow as tf
```


```python
tf.__version__
```




    '2.11.0'



## Data Preprocessing

### Importing the Dataset


```python
dataset = pd.read_excel("./Expenditure-churn (3).xlsx")
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 12 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       10000 non-null  int64  
     1   gender    10000 non-null  int64  
     2   marital   10000 non-null  float64
     3   dep       10000 non-null  int64  
     4   Income    10000 non-null  float64
     5   Job yrs   10000 non-null  int64  
     6   Town yrs  10000 non-null  int64  
     7   Yrs Ed    10000 non-null  int64  
     8   Dri Lic   10000 non-null  int64  
     9   Own Home  10000 non-null  int64  
     10  # Cred C  10000 non-null  int64  
     11  Churn     10000 non-null  int64  
    dtypes: float64(2), int64(10)
    memory usage: 937.6 KB
    


```python
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
```

### Splitting the data into Training and Test set


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

### Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Building the ANN

### Initializing the ann


```python
ann = tf.keras.models.Sequential()
```

### Adding the input layer and the first hidden layer  


```python
ann.add(tf.keras.layers.Dense(units=64, activation='relu')) 
```

### Adding the second hidden layer


```python
ann.add(tf.keras.layers.Dense(units=48, activation='relu'))
```

### Adding the Output layer


```python
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

## Training the ANN

### Compiling the ANN


```python
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

### Training the ANN on the Training Set


```python
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

    Epoch 1/100
    235/235 [==============================] - 3s 4ms/step - loss: 0.2616 - accuracy: 0.8872
    Epoch 2/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0635 - accuracy: 0.9825
    Epoch 3/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0398 - accuracy: 0.9871
    Epoch 4/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0300 - accuracy: 0.9904
    Epoch 5/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0234 - accuracy: 0.9927
    Epoch 6/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0192 - accuracy: 0.9943
    Epoch 7/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0166 - accuracy: 0.9956
    Epoch 8/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0144 - accuracy: 0.9959
    Epoch 9/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0126 - accuracy: 0.9968
    Epoch 10/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0114 - accuracy: 0.9961
    Epoch 11/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0106 - accuracy: 0.9968
    Epoch 12/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0087 - accuracy: 0.9977
    Epoch 13/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0102 - accuracy: 0.9965
    Epoch 14/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0075 - accuracy: 0.9977
    Epoch 15/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0064 - accuracy: 0.9980
    Epoch 16/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0067 - accuracy: 0.9984
    Epoch 17/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0053 - accuracy: 0.9988
    Epoch 18/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0045 - accuracy: 0.9991
    Epoch 19/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0041 - accuracy: 0.9989
    Epoch 20/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0065 - accuracy: 0.9975
    Epoch 21/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0048 - accuracy: 0.9987
    Epoch 22/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0038 - accuracy: 0.9988
    Epoch 23/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0026 - accuracy: 0.9995
    Epoch 24/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0043 - accuracy: 0.9984
    Epoch 25/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0020 - accuracy: 0.9999
    Epoch 26/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0023 - accuracy: 0.9996
    Epoch 27/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0032 - accuracy: 0.9989
    Epoch 28/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0048 - accuracy: 0.9984
    Epoch 29/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0050 - accuracy: 0.9981
    Epoch 30/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9997
    Epoch 31/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9996
    Epoch 32/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9997
    Epoch 33/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0016 - accuracy: 0.9999
    Epoch 34/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0022 - accuracy: 0.9996
    Epoch 35/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0023 - accuracy: 0.9993
    Epoch 36/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0069 - accuracy: 0.9975
    Epoch 37/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0043 - accuracy: 0.9981
    Epoch 38/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9996
    Epoch 39/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.0494e-04 - accuracy: 0.9999
    Epoch 40/100
    235/235 [==============================] - 1s 3ms/step - loss: 5.9393e-04 - accuracy: 1.0000
    Epoch 41/100
    235/235 [==============================] - 1s 3ms/step - loss: 5.7110e-04 - accuracy: 1.0000
    Epoch 42/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0029 - accuracy: 0.9987
    Epoch 43/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0028 - accuracy: 0.9992
    Epoch 44/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0015 - accuracy: 0.9995
    Epoch 45/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.0946e-04 - accuracy: 1.0000
    Epoch 46/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.3892e-04 - accuracy: 1.0000
    Epoch 47/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.9667e-04 - accuracy: 1.0000
    Epoch 48/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.2670e-04 - accuracy: 1.0000
    Epoch 49/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0066 - accuracy: 0.9981
    Epoch 50/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0062 - accuracy: 0.9981
    Epoch 51/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - accuracy: 0.9988
    Epoch 52/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.7653e-04 - accuracy: 1.0000
    Epoch 53/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.1398e-04 - accuracy: 1.0000
    Epoch 54/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.4859e-04 - accuracy: 1.0000
    Epoch 55/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.7687e-04 - accuracy: 1.0000
    Epoch 56/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.8085e-04 - accuracy: 1.0000
    Epoch 57/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.9511e-04 - accuracy: 1.0000
    Epoch 58/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.7072e-04 - accuracy: 1.0000
    Epoch 59/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.6152e-04 - accuracy: 1.0000
    Epoch 60/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.6068e-04 - accuracy: 1.0000
    Epoch 61/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0107 - accuracy: 0.9973
    Epoch 62/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0038 - accuracy: 0.9981
    Epoch 63/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.4746e-04 - accuracy: 1.0000
    Epoch 64/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.5776e-04 - accuracy: 1.0000
    Epoch 65/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.0422e-04 - accuracy: 1.0000
    Epoch 66/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.5546e-04 - accuracy: 1.0000
    Epoch 67/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.5003e-04 - accuracy: 1.0000
    Epoch 68/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.3928e-04 - accuracy: 1.0000
    Epoch 69/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.3602e-04 - accuracy: 1.0000
    Epoch 70/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.8484e-04 - accuracy: 1.0000
    Epoch 71/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.6412e-04 - accuracy: 0.9999
    Epoch 72/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0033 - accuracy: 0.9987
    Epoch 73/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0093 - accuracy: 0.9976
    Epoch 74/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0022 - accuracy: 0.9993
    Epoch 75/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.5701e-04 - accuracy: 1.0000
    Epoch 76/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.2967e-04 - accuracy: 1.0000
    Epoch 77/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.9854e-05 - accuracy: 1.0000
    Epoch 78/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.9170e-05 - accuracy: 1.0000
    Epoch 79/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.3809e-05 - accuracy: 1.0000
    Epoch 80/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.6040e-05 - accuracy: 1.0000
    Epoch 81/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0550e-04 - accuracy: 1.0000
    Epoch 82/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0744e-04 - accuracy: 1.0000
    Epoch 83/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.3892e-04 - accuracy: 1.0000
    Epoch 84/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0314e-04 - accuracy: 1.0000
    Epoch 85/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0160 - accuracy: 0.9975
    Epoch 86/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0038 - accuracy: 0.9985
    Epoch 87/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0020 - accuracy: 0.9991
    Epoch 88/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0037 - accuracy: 0.9987
    Epoch 89/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.8882e-04 - accuracy: 1.0000
    Epoch 90/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.2288e-04 - accuracy: 1.0000
    Epoch 91/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.1115e-04 - accuracy: 1.0000
    Epoch 92/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.5189e-05 - accuracy: 1.0000
    Epoch 93/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0408e-04 - accuracy: 1.0000
    Epoch 94/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.0245e-05 - accuracy: 1.0000
    Epoch 95/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.1883e-05 - accuracy: 1.0000
    Epoch 96/100
    235/235 [==============================] - 1s 3ms/step - loss: 7.3396e-05 - accuracy: 1.0000
    Epoch 97/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.4839e-05 - accuracy: 1.0000
    Epoch 98/100
    235/235 [==============================] - 1s 3ms/step - loss: 7.1007e-05 - accuracy: 1.0000
    Epoch 99/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.0511e-05 - accuracy: 1.0000
    Epoch 100/100
    235/235 [==============================] - 1s 3ms/step - loss: 6.7545e-05 - accuracy: 1.0000
    




    <keras.callbacks.History at 0x1b04584f788>



## Making Predictions and evaluating the model

### Predicting the test set results


```python
y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

    79/79 [==============================] - 0s 2ms/step
    [[0 0]
     [0 0]
     [0 0]
     ...
     [0 0]
     [0 0]
     [0 0]]
    

### Making the Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
cm = confusion_matrix(y_test, y_pred)
# Print the errors
print("Accuracy Score:   "+str(accuracy_score(y_pred, y_test)*100))
print("Precision Score:  "+str(precision_score(y_pred, y_test)*100))
print("Recall Score:     "+str(recall_score(y_pred, y_test)*100))
print("roc_auc_score:    "+str(accuracy_score(y_pred, y_test)*100))
print("\nConfusion Matrix:\n", confusion_matrix(y_pred, y_test))
```

    Accuracy Score:   99.6
    Precision Score:  99.26650366748166
    Recall Score:     99.50980392156863
    roc_auc_score:    99.6
    
    Confusion Matrix:
     [[1678    6]
     [   4  812]]
    
# ANN Classification

### Importing the libraries


```python
import numpy as np
import pandas as pd
import tensorflow as tf
```


```python
tf.__version__
```




    '2.11.0'



## Data Preprocessing

### Importing the Dataset


```python
dataset = pd.read_excel("./Expenditure-churn (3).xlsx")
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 12 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       10000 non-null  int64  
     1   gender    10000 non-null  int64  
     2   marital   10000 non-null  float64
     3   dep       10000 non-null  int64  
     4   Income    10000 non-null  float64
     5   Job yrs   10000 non-null  int64  
     6   Town yrs  10000 non-null  int64  
     7   Yrs Ed    10000 non-null  int64  
     8   Dri Lic   10000 non-null  int64  
     9   Own Home  10000 non-null  int64  
     10  # Cred C  10000 non-null  int64  
     11  Churn     10000 non-null  int64  
    dtypes: float64(2), int64(10)
    memory usage: 937.6 KB
    


```python
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
```

### Splitting the data into Training and Test set


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

### Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Building the ANN

### Initializing the ann


```python
ann = tf.keras.models.Sequential()
```

### Adding the input layer and the first hidden layer  


```python
ann.add(tf.keras.layers.Dense(units=64, activation='relu')) 
```

### Adding the second hidden layer


```python
ann.add(tf.keras.layers.Dense(units=48, activation='relu'))
```

### Adding the Output layer


```python
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

## Training the ANN

### Compiling the ANN


```python
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

### Training the ANN on the Training Set


```python
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

    Epoch 1/100
    235/235 [==============================] - 3s 4ms/step - loss: 0.2616 - accuracy: 0.8872
    Epoch 2/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0635 - accuracy: 0.9825
    Epoch 3/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0398 - accuracy: 0.9871
    Epoch 4/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0300 - accuracy: 0.9904
    Epoch 5/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0234 - accuracy: 0.9927
    Epoch 6/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0192 - accuracy: 0.9943
    Epoch 7/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0166 - accuracy: 0.9956
    Epoch 8/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0144 - accuracy: 0.9959
    Epoch 9/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0126 - accuracy: 0.9968
    Epoch 10/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0114 - accuracy: 0.9961
    Epoch 11/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0106 - accuracy: 0.9968
    Epoch 12/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0087 - accuracy: 0.9977
    Epoch 13/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0102 - accuracy: 0.9965
    Epoch 14/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0075 - accuracy: 0.9977
    Epoch 15/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0064 - accuracy: 0.9980
    Epoch 16/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0067 - accuracy: 0.9984
    Epoch 17/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0053 - accuracy: 0.9988
    Epoch 18/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0045 - accuracy: 0.9991
    Epoch 19/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0041 - accuracy: 0.9989
    Epoch 20/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0065 - accuracy: 0.9975
    Epoch 21/100
    235/235 [==============================] - 1s 4ms/step - loss: 0.0048 - accuracy: 0.9987
    Epoch 22/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0038 - accuracy: 0.9988
    Epoch 23/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0026 - accuracy: 0.9995
    Epoch 24/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0043 - accuracy: 0.9984
    Epoch 25/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0020 - accuracy: 0.9999
    Epoch 26/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0023 - accuracy: 0.9996
    Epoch 27/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0032 - accuracy: 0.9989
    Epoch 28/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0048 - accuracy: 0.9984
    Epoch 29/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0050 - accuracy: 0.9981
    Epoch 30/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9997
    Epoch 31/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9996
    Epoch 32/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9997
    Epoch 33/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0016 - accuracy: 0.9999
    Epoch 34/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0022 - accuracy: 0.9996
    Epoch 35/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0023 - accuracy: 0.9993
    Epoch 36/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0069 - accuracy: 0.9975
    Epoch 37/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0043 - accuracy: 0.9981
    Epoch 38/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0018 - accuracy: 0.9996
    Epoch 39/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.0494e-04 - accuracy: 0.9999
    Epoch 40/100
    235/235 [==============================] - 1s 3ms/step - loss: 5.9393e-04 - accuracy: 1.0000
    Epoch 41/100
    235/235 [==============================] - 1s 3ms/step - loss: 5.7110e-04 - accuracy: 1.0000
    Epoch 42/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0029 - accuracy: 0.9987
    Epoch 43/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0028 - accuracy: 0.9992
    Epoch 44/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0015 - accuracy: 0.9995
    Epoch 45/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.0946e-04 - accuracy: 1.0000
    Epoch 46/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.3892e-04 - accuracy: 1.0000
    Epoch 47/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.9667e-04 - accuracy: 1.0000
    Epoch 48/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.2670e-04 - accuracy: 1.0000
    Epoch 49/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0066 - accuracy: 0.9981
    Epoch 50/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0062 - accuracy: 0.9981
    Epoch 51/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - accuracy: 0.9988
    Epoch 52/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.7653e-04 - accuracy: 1.0000
    Epoch 53/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.1398e-04 - accuracy: 1.0000
    Epoch 54/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.4859e-04 - accuracy: 1.0000
    Epoch 55/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.7687e-04 - accuracy: 1.0000
    Epoch 56/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.8085e-04 - accuracy: 1.0000
    Epoch 57/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.9511e-04 - accuracy: 1.0000
    Epoch 58/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.7072e-04 - accuracy: 1.0000
    Epoch 59/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.6152e-04 - accuracy: 1.0000
    Epoch 60/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.6068e-04 - accuracy: 1.0000
    Epoch 61/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0107 - accuracy: 0.9973
    Epoch 62/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0038 - accuracy: 0.9981
    Epoch 63/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.4746e-04 - accuracy: 1.0000
    Epoch 64/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.5776e-04 - accuracy: 1.0000
    Epoch 65/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.0422e-04 - accuracy: 1.0000
    Epoch 66/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.5546e-04 - accuracy: 1.0000
    Epoch 67/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.5003e-04 - accuracy: 1.0000
    Epoch 68/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.3928e-04 - accuracy: 1.0000
    Epoch 69/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.3602e-04 - accuracy: 1.0000
    Epoch 70/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.8484e-04 - accuracy: 1.0000
    Epoch 71/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.6412e-04 - accuracy: 0.9999
    Epoch 72/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0033 - accuracy: 0.9987
    Epoch 73/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0093 - accuracy: 0.9976
    Epoch 74/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0022 - accuracy: 0.9993
    Epoch 75/100
    235/235 [==============================] - 1s 3ms/step - loss: 3.5701e-04 - accuracy: 1.0000
    Epoch 76/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.2967e-04 - accuracy: 1.0000
    Epoch 77/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.9854e-05 - accuracy: 1.0000
    Epoch 78/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.9170e-05 - accuracy: 1.0000
    Epoch 79/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.3809e-05 - accuracy: 1.0000
    Epoch 80/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.6040e-05 - accuracy: 1.0000
    Epoch 81/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0550e-04 - accuracy: 1.0000
    Epoch 82/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0744e-04 - accuracy: 1.0000
    Epoch 83/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.3892e-04 - accuracy: 1.0000
    Epoch 84/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0314e-04 - accuracy: 1.0000
    Epoch 85/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0160 - accuracy: 0.9975
    Epoch 86/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0038 - accuracy: 0.9985
    Epoch 87/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0020 - accuracy: 0.9991
    Epoch 88/100
    235/235 [==============================] - 1s 3ms/step - loss: 0.0037 - accuracy: 0.9987
    Epoch 89/100
    235/235 [==============================] - 1s 3ms/step - loss: 4.8882e-04 - accuracy: 1.0000
    Epoch 90/100
    235/235 [==============================] - 1s 3ms/step - loss: 2.2288e-04 - accuracy: 1.0000
    Epoch 91/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.1115e-04 - accuracy: 1.0000
    Epoch 92/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.5189e-05 - accuracy: 1.0000
    Epoch 93/100
    235/235 [==============================] - 1s 3ms/step - loss: 1.0408e-04 - accuracy: 1.0000
    Epoch 94/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.0245e-05 - accuracy: 1.0000
    Epoch 95/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.1883e-05 - accuracy: 1.0000
    Epoch 96/100
    235/235 [==============================] - 1s 3ms/step - loss: 7.3396e-05 - accuracy: 1.0000
    Epoch 97/100
    235/235 [==============================] - 1s 3ms/step - loss: 9.4839e-05 - accuracy: 1.0000
    Epoch 98/100
    235/235 [==============================] - 1s 3ms/step - loss: 7.1007e-05 - accuracy: 1.0000
    Epoch 99/100
    235/235 [==============================] - 1s 3ms/step - loss: 8.0511e-05 - accuracy: 1.0000
    Epoch 100/100
    235/235 [==============================] - 1s 3ms/step - loss: 6.7545e-05 - accuracy: 1.0000
    




    <keras.callbacks.History at 0x1b04584f788>



## Making Predictions and evaluating the model

### Predicting the test set results


```python
y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

    79/79 [==============================] - 0s 2ms/step
    [[0 0]
     [0 0]
     [0 0]
     ...
     [0 0]
     [0 0]
     [0 0]]
    

### Making the Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
cm = confusion_matrix(y_test, y_pred)
# Print the errors
print("Accuracy Score:   "+str(accuracy_score(y_pred, y_test)*100))
print("Precision Score:  "+str(precision_score(y_pred, y_test)*100))
print("Recall Score:     "+str(recall_score(y_pred, y_test)*100))
print("roc_auc_score:    "+str(accuracy_score(y_pred, y_test)*100))
print("\nConfusion Matrix:\n", confusion_matrix(y_pred, y_test))
```

    Accuracy Score:   99.6
    Precision Score:  99.26650366748166
    Recall Score:     99.50980392156863
    roc_auc_score:    99.6
    
    Confusion Matrix:
     [[1678    6]
     [   4  812]]
    
