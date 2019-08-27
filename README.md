# Medical-Appointment-No-Shows
This analysis is part of the Tebian Camp and aims to explore a dataset containing approximately 100k medical appointments from the Brazilian public health system.

## Introduction
This analysis will use the no-show appointments dataset which collects information from 100k medical appointments and is focused on the question of whether or not patients show up for their appointment.
Many times people do not show up for a medical appointment. No-show is a loss for doctors since they lose their time. On the other hand, patients who wanted an appointment as soon as possible were unable to get one.
Thus, there are two losses: the time loss for the doctor and the loss of an appointment for the person in need. The analysis is focused on finding which affects patients to show or not show up to appointments.

## Understanding the data
We use dataset Medical Appointment No Shows from [Kaggle](https://www.kaggle.com/joniarroba/noshowappointments).
## Dataset Description
| Name | Value(s)  | Description  |
| ------- | --- | --- |
|PatienID |	number|	identification of a patient|
|AppointmentID|	number|	identification of each appointment|
|Gender	|F or M	|it says 'F' if female or 'M' if man|
|ScheduledDay|	date|	tells us on what day the patient set up their appointment|
|AppointmentDay|	date|	the day of the actuall appointment, when they have to visit the doctor|
|Age|	number|	how old is the patient|
|Neighbourhood|	string	|indicates the location of the hospital|
|Scholarship|	0 or 1|	indicates whether or not the patient is enrolled in scholarship|
|Hipertension|	0 or 1|	indicates if the patient has hipertension|
|Diabetes|	0 or 1|	indicates if the patient has diabetes|
|Alcoholism|	0 or 1|	indicates if the patient is an alcoholic|
|Handcap|	0 or 1|	indicates if the patient is handicaped|
|SMS_received|	0 or 1|	1 or more messages sent to the patient|
|No-show|	Yes or No|	it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up|

## Exploratory data Analysis
![image](https://user-images.githubusercontent.com/27751735/63810179-cc02f280-c92c-11e9-8dc0-9fd13a465edf.png)
<br>
Age: There are many young people in the dataset, but in general the number of patients down for older than 60 years.

![image](https://user-images.githubusercontent.com/27751735/63810189-d1f8d380-c92c-11e9-8e10-db95fbed9066.png)
<br>
Alcoholism: Most of the patients are not alcoholics.

![image](https://user-images.githubusercontent.com/27751735/63810195-d624f100-c92c-11e9-99a7-85c6bee0864e.png)
<br>
Diabetes: Most of the patients are not diabetes.

![image](https://user-images.githubusercontent.com/27751735/63810662-f30df400-c92d-11e9-8f82-510a57f4a3d3.png)
<br>
Handicap: most of the people not handicapped.

![image](https://user-images.githubusercontent.com/27751735/63810672-f86b3e80-c92d-11e9-9009-22a9fd1d100d.png)
<br>
Hypertension: Most patients do not have hypertension.

## Questions to be Explored
- How many percent of patients missed their scheduled appointment?
![image](https://user-images.githubusercontent.com/27751735/63810881-919a5500-c92e-11e9-8727-72b18daacbf7.png)<br> 
20% of appointments were missed. 

- What is the gender distribution for show / no-show patients?<br>
![image](https://user-images.githubusercontent.com/27751735/63810887-95c67280-c92e-11e9-97a1-d86cc5479e3e.png)<br>
Out of 71831 appointments made by females, 14588 were missed with the ratio of 20%. <br>
Out of 38685 appointments made by males, 7723 were missed with the ratio of 20%.

- Important factors to know in order to predict if a patient will show up for their scheduled appointment<br>
![image](https://user-images.githubusercontent.com/27751735/63810903-a2e36180-c92e-11e9-90ea-2f2c47c67270.png)
![image](https://user-images.githubusercontent.com/27751735/63810908-a7a81580-c92e-11e9-8955-011e64d56dcf.png)
![image](https://user-images.githubusercontent.com/27751735/63810916-abd43300-c92e-11e9-823a-309af28213a9.png)
![image](https://user-images.githubusercontent.com/27751735/63810921-af67ba00-c92e-11e9-9174-a1ae20d12aca.png)<br>
For all features the distributions of show / no-show for different categories look very similar. There is no clear indication of any of these variables having bigger then others impact on show / no-show characteristics. The charts confirm about 20% no-show rate for most categories.

## Data Analysis
The Models which used are:
* Decision Tree Classifier
* Logistic Regression
* Random Forest Classifier
* GaussianNB
* SGD Classifier
* SVC

The Metric which used are:
* accuracy score
* recall score
* precision score
* zero one loss

The Feature selection which used is:
* KNeighbors Classifier

## The Source code of Models
#### Import data analysis packages:
```
# Data analysis packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import classes from library
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split from sklearn.feature_selection import SelectKBest
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import zero_one_loss
from sklearn import model_selection
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression from sklearn.ensemble import RandomForestClassifier from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```
#### Load dataset:
```
#load dataset
dataset = pd.read_csv('medicalAppointment.csv')
dataset.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day', 'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']
#determine just 10K row to work on it
data=dataset.head(10000)
```

## Data Cleaning
* Fixed column misspellings, incorporated underscores:
```
dataset.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day', 'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']
```
* Fix columns with missing value:
```
data['scholarship'] = np.where(data['scholarship']>0, 1, 0)
data['hypertension'] = np.where(data['hypertension']>0, 1, 0)
data['alcoholism'] = np.where(data['alcoholism']>0, 1, 0)
data['handicap'] = np.where(data['handicap']>0, 1, 0)
data['sms_received'] = np.where(data['sms_received']>0, 1, 0)
```
* Checking the Age attribute:
```
data[data['age']<0]
```
* Checking the Appointment id is unique:
```
data['appointment_id'] = data['appointment_id'].apply(lambda x: str(int(x)));
len(data['appointment_id'].unique())
```
* Gender attribute:
```
data['gender']=x.gender[x.gender=='M']=0
data['gender']=x.gender[x.gender=='F']=1
```
* Check all attributes have same number of values:
```
dataset.info()
```
![image](https://user-images.githubusercontent.com/27751735/63812166-336f7100-c932-11e9-8ec1-ddd1cd214f60.png)

* Converts the type of scheduled_day and appointment_day to datetime:
```
data['scheduled_day'] = pd.to_datetime(data['scheduled_day'])
data['appointment_day'] = pd.to_datetime(data['appointment_day'])
```
* Create a variable called waiting_time:
```
data['waiting_time'] = data["appointment_day"].sub(data["scheduled_day"], axis=0)
data["waiting_time"] = (data["waiting_time"] / np.timedelta64(1, 'D')).abs()
```
* Drop unused columns:
```
x.drop(x.columns[[0,1,3,4,6]],axis=1, inplace=True )
```

## After cleaning: 

```
#split data to train and test
x_train, x_test1, y_train , y_test = train_test_split(x , y, test_size=0.2 , random_state = 4)

#select best features and result depend on the value of k
H = SelectKBest(chi2 , k=7)
x_new = H.fit_transform(x_train , y_train) print([H.get_support(indices=True)])

#drop unused columns 
x_train.drop(x_train.columns[[3,4]],axis=1, inplace=True ) Kfold = KFold(n_splits=10 , random_state=7)

#chosen model
model1 =LogisticRegression(solver='lbfgs' ,multi_class='ovr') model2 =RandomForestClassifier()
model3 =GaussianNB()
model4=SGDClassifier()
model5=SVC()
model6= DecisionTreeClassifier()
acores=[]
acores1=[] models=[model1,model2,model3,model4,model5,model6] 

#loop for find the predict of each model 
for m in models:
  cv_result= model_selection.cross_val_score(m , x_train , y_train , cv=Kfold , scoring='accuracy')
  msg =cv_result.mean()
  acores.append(msg)

#print the result of estimated performance for each model and represented by diagram
print("Estimated performance ",acores)
k_range = list(range(1,7))
plt.scatter(k_range ,acores )

#drop unused columns x_test1.drop(x_test1.columns[[3,4]],axis=1, inplace=True ) x1=0
predictions=0

#test the model and print accuracy_score,recall_score,precision_score and zero_one_loss for each model
for m in models:
  x1=x1+1
  m.fit(x_train , y_train)
  predictions = m.predict(x_test1)
  print( "%s: %f" % ('\n accuracy_score: ' , accuracy_score(y_test , predictions)) )
  print( "%s: %f" % ('\n recall_score: ' , recall_score(y_test , predictions,average='micro')) ) print( "%s: %f" % ('\n precision_score: ' , precision_score(y_test , predictions,average='micro')) )
  print( "%s: %f" % ('\n zero_one_loss: ' , zero_one_loss(y_test , predictions,normalize=True)) ) acores1.append(accuracy_score(y_test , predictions)) 
  print("#############################",x1) 
  predictions = 0

#print the result of real performance for each model and represented by diagram
print(acores1) plt.scatter(k_range ,acores1 )
```

## Improvement performance
Every time we choose the beast feature depends on the value of key


- When K = 1:<br>
![image](https://user-images.githubusercontent.com/27751735/63812680-cc52bc00-c933-11e9-828e-0e8d173058a1.png)<br>
- When K = 2:<br>
![image](https://user-images.githubusercontent.com/27751735/63812688-d07ed980-c933-11e9-8683-a07d0fca261f.png)<br>
- When K = 3:<br>
![image](https://user-images.githubusercontent.com/27751735/63812693-d8d71480-c933-11e9-8435-36ab01788359.png)<br>
- When K = 4:<br>
![image](https://user-images.githubusercontent.com/27751735/63812967-c01b2e80-c934-11e9-90bb-35522584a7aa.png)<br>
- When K = 5:<br>
![image](https://user-images.githubusercontent.com/27751735/63812713-e391a980-c933-11e9-9c31-d81a0b47b89e.png)<br>
- When K = 6:<br>
![image](https://user-images.githubusercontent.com/27751735/63812721-eb514e00-c933-11e9-889c-8cfbcfac3e94.png)<br>
- When K = 7:<br>
![image](https://user-images.githubusercontent.com/27751735/63812731-f3a98900-c933-11e9-903f-b5e850ccbff3.png)<br>
- When K = 8:<br>
![image](https://user-images.githubusercontent.com/27751735/63812734-f86e3d00-c933-11e9-95cf-1221d59588bf.png)<br>

## Best Performance
| Model | preformance  | 
| ------- | --- |	
|GaussianNB	|80%|	
|Decision Tree Classifier	|	79%|	
|SGD Classifier	|	79%|	
|Random Forest Classifier	|	78%|	
|Logistic Regression	|	73%|	
|SVC	|	72%|	

## Conclusion
In this report, we are focus on the accuracy of prediction. Our goal is improve performance. We identified the machine learning algorithm for the current dataset; therefore, we compare different algorithms and identified the best performing algorithms.
