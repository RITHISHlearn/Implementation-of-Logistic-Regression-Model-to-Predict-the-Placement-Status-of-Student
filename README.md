# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RITHISH P
RegisterNumber:  212223230173
*/
```
import pandas as pd

data=pd.read_csv("Placement_Data.csv")

data.head()


data1=data.copy()

data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column

data1.head()


data1.isnull().sum()


data1.duplicated().sum()


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])

data1["ssc_b"]=le.fit_transform(data1["ssc_b"])

data1["hsc_b"]=le.fit_transform(data1["hsc_b"])

data1["hsc_s"]=le.fit_transform(data1["hsc_s"])

data1["degree_t"]=le.fit_transform(data1["degree_t"])

data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"] ) 

data1["status"]=le.fit_transform(data1["status"])  

data1 


x=data1.iloc[:,:-1]

x


y=data1["status"]

y


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver="liblinear")

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_pred


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)

accuracy


from sklearn.metrics import confusion_matrix

confusion=confusion_matrix(y_test,y_pred)

confusion


from sklearn.metrics import classification_report

classification_report1 = classification_report(y_test,y_pred)

print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:
##Placement Data:
![11](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/54d6318c-b56b-4945-a437-1025391f996e)


##Salary Data:
![10](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/d3b269e9-e6fd-47c5-a7fe-23c1fe6b5678)


##Checking the null() function:
![9](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/40624c89-518f-4012-976d-76444cc9fb93)


##Data Duplicate:
![8](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/474f5886-52ec-4014-b470-a7f32e914d83)


##Print Data:
![7](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/1dcd41d8-0a43-4b64-a598-396c3533789f)


##Data-Status:
![6](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/abc280e0-a0b3-4d8f-b08b-495bccde7ad3)


##Y_prediction array:
![5](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/f05e3d4b-4d2e-4b11-a687-8c48b51875ca)


##Accuracy value:
![4](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/a13468c1-2a8a-481b-a35c-0c0826aacf6d)


##Confusion array:
![3](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/2059f51d-ae22-4c38-a049-053a613c0f74)


##Classification Report:
![2](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/6dc479b3-c917-4745-b51a-1294ded36b3a)


##Prediction of LR:
![1](https://github.com/RITHISHlearn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446645/94ea2e05-6639-4f11-ae88-463505d815d6)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
