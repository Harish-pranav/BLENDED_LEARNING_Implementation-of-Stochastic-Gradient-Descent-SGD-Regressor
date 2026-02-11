# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
4. Split the data into features (X) and target (y).
5. Divide the data into training and testing sets. 6.Create an SGD Regressor model. 7.Fit the model on the training data. 8.Evaluate the model performance. 9.Make predictions and visualize the results.

## Program:
```py
/*
Program to implement SGD Regressor for linear regression.
Developed by: Harish Pranav
RegisterNumber:  212225040117
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train.ravel())
y_pred = sgd_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Name: Harish Pranav')
print('Reg. No: 212225040117')
print("="*50)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print("="*50)
print("\nModel Coefficients:")
print("\nCoefficients:", sgd_model.coef_)
print("\nIntercept:", sgd_model.intercept_)
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(),y.max()],'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual VS Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()
```

## Output:
<img width="1387" height="630" alt="image" src="https://github.com/user-attachments/assets/1ea2d43f-e988-422d-b3b7-bb58be59f636" />
<img width="1341" height="733" alt="image" src="https://github.com/user-attachments/assets/a99428a2-b969-4e68-aea4-35af04160774" />
<img width="1409" height="439" alt="image" src="https://github.com/user-attachments/assets/6aabf3ee-1e9e-4d2c-a9dd-eabee45223f8" />
<img width="1376" height="570" alt="image" src="https://github.com/user-attachments/assets/84f7f318-3448-4160-b4ca-2940ca8a1f17" />




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
