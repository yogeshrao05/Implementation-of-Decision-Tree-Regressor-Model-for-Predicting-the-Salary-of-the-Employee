# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Yogesh rao S D
RegisterNumber: 212222110055
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:
### Data Head:
![Screenshot 2024-04-02 094141](https://github.com/amal-2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148410730/92e9d8f7-78b7-41f4-b441-edb4279d8f0d)


### Data Info:
![Screenshot 2024-04-02 094246](https://github.com/amal-2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148410730/611c2f84-09e5-4040-9ec8-f7e2c08175a8)


### isnull() sum():
![Screenshot 2024-04-02 094333](https://github.com/amal-2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148410730/1f0c3c0c-5949-4a80-a74b-639171f3fa5a)


### Data Head for salary:
![Screenshot 2024-04-02 094423](https://github.com/amal-2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148410730/dcba8525-df04-48ef-a97d-faf2b103c6a1)


### Mean Squared Error:
![Screenshot 2024-04-02 094511](https://github.com/amal-2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148410730/f51e8787-8412-41da-b05d-b94a52ab5437)
  

### r2 Value:
![Screenshot 2024-04-02 095714](https://github.com/amal-2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148410730/c5e08f65-efd0-487f-8b20-e236dc993fd2)


### Data Prediction:
![Screenshot 2024-04-02 095745](https://github.com/amal-2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148410730/3181b25d-30f0-4d1b-8087-0ec315cec41c)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
