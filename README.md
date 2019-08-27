# Medical-Appointment-No-Shows
This analysis is part of the Tebian Camp and aims to explore a dataset containing approximately 100k medical appointments from the Brazilian public health system.

# Introduction
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

# Exploratory data Analysis
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

# Data Analysis
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
