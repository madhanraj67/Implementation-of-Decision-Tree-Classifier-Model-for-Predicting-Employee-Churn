# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Importing the necessary libraries, primarily pandas for data manipulation and sklearn for machine learning functionalities.
2. The dataset is divided into training and testing sets using train_test_split with a test size of 20% and a random state of 100 to ensure reproducibility.
3. A DecisionTreeClassifier is instantiated with the criterion set to "entropy", which measures the quality of a split.
4. The accuracy of the model is computed by comparing the predicted values with the actual values using metrics.accuracy_score(Y_test, Y_pred).
5. Predict churn for an employee with specific features using dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]]).
6. The output will be either 0 (the employee will not leave) or 1 (the employee will leave).
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:MADHANRAJ P
RegisterNumber:212223220052
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
X=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
Y=data["left"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## data.head()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/46750aa6-b43f-4279-a69b-d0a2a253c68a)
## data.info()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/6c62b143-b7bc-476b-ad4d-5335b6553410)
## isnull() and sum()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/bd038511-15d5-4250-b3ad-2e3bbd67aa16)
## data value counts()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/434d12ee-0a71-49bb-8313-ff117eba454e)
## data.head() for salary
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/4e931b84-cbae-4f29-93b0-02e0f0891fd9)
## x.head()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/14328c18-e8be-424a-ac64-60a944491a0e)
## accuracy value()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/cf46dd5e-d540-4ec0-8443-e392d67df61c)
## data prediction
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113915622/aa576ae0-394e-4c34-b3e8-3c4710fe2249)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
