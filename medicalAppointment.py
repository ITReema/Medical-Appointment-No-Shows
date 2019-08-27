# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#import classes from library 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import  RandomForestClassifier 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
#load dataset
dataset = pd.read_csv('medicalAppointment.csv') 

#changes names of columns 
dataset.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day', 
              'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',
              'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']

#determine just 10 000 row to work on it 
data=dataset.head(10000)

########################################################Cleaning Data #########################################################
#check data information 
print(data.info())
#make sure the specific columns contain only 1 or 0 
data['scholarship'] = np.where(data['scholarship']>0, 1, 0)
data['hypertension'] = np.where(data['hypertension']>0, 1, 0)
data['alcoholism'] = np.where(data['alcoholism']>0, 1, 0)
data['handicap'] = np.where(data['handicap']>0, 1, 0)
data['sms_received'] = np.where(data['sms_received']>0, 1, 0)
data[data['age']<0]

# Converts the type of two variables to datetime 
data['scheduled_day'] = pd.to_datetime(data['scheduled_day'])
data['appointment_day'] = pd.to_datetime(data['appointment_day'])

# Create a variable called "waiting_time" by subtracting the date the patient made the appointment and the date of the appointment.
data['waiting_time'] = data["appointment_day"].sub(data["scheduled_day"], axis=0)

# Convert the result "waiting_time" to number of days between appointment day and scheduled day. 
data["waiting_time"] = (data["waiting_time"] / np.timedelta64(1, 'D')).abs()


#make sure all appointment is unique
data['appointment_id'] = data['appointment_id'].apply(lambda x: str(int(x)));
len(data['appointment_id'].unique())

####################################################### Working on Columns ##################################################################

#choose specific columns
x=data[['patient_id', 'appointment_id', 'gender', 'scheduled_day', 
              'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',
              'diabetes', 'alcoholism', 'handicap', 'sms_received','waiting_time']]
#class label
y=data[['no_show']]

#convert M to 0 and F to 1
data['gender']=x.gender[x.gender=='M']=0
data['gender']=x.gender[x.gender=='F']=1

#drop that columns (patient_id,appointment_id,scheduled_day,appointment_day,neighbourhood)
x.drop(x.columns[[0,1,3,4,6]],axis=1, inplace=True )

####################################################### Modelling of Dataset #########################################################

#split data to train and test 
x_train, x_test1, y_train , y_test = train_test_split(x , y, test_size=0.2 , random_state = 4)

#select  best features and result depend on the value of k 
H= SelectKBest(chi2 , k=7)
x_new = H.fit_transform(x_train , y_train)
print([H.get_support(indices=True)])
#drop unused columns 
x_train.drop(x_train.columns[[3,4]],axis=1, inplace=True ) 
Kfold = KFold(n_splits=10 , random_state=7)
#chosen model 
model1 =LogisticRegression(solver='lbfgs' ,multi_class='ovr')
model2 =RandomForestClassifier()
model3 =GaussianNB()
model4=SGDClassifier()
model5=SVC()
model6= DecisionTreeClassifier()

acores=[]
acores1=[]
models=[model1,model2,model3,model4,model5,model6]
#loop for find the predict of each model 
for m in models:
        cv_result= model_selection.cross_val_score(m , x_train , y_train , cv=Kfold , scoring='accuracy')
        msg =cv_result.mean() 
        acores.append(msg)
#print the result of estimated performance for each model and represented by diagram    
print("Estimated performance ",acores)
k_range = list(range(1,7))
plt.scatter(k_range ,acores )
#drop unused columns   
x_test1.drop(x_test1.columns[[3,4]],axis=1, inplace=True )
x1=0
predictions=0
##test the model and print accuracy_score,recall_score,precision_score and zero_one_loss for each model
for m in models: 
    x1=x1+1     
    m.fit(x_train , y_train)
    predictions = m.predict(x_test1)
    print( "%s: %f" % ('\n accuracy_score: ' , accuracy_score(y_test , predictions)) )
    print( "%s: %f" % ('\n recall_score: ' ,  recall_score(y_test , predictions,average='micro')) )
    print( "%s: %f" % ('\n precision_score: ' , precision_score(y_test , predictions,average='micro')) )
    print( "%s: %f" % ('\n zero_one_loss: ' , zero_one_loss(y_test , predictions,normalize=True)) )
    acores1.append(accuracy_score(y_test , predictions))
    print("#############################",x1)
    predictions = 0
#print the result of real performance for each model and represented by diagram       
print(acores1)
plt.scatter(k_range ,acores1 )