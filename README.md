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
![image](https://user-images.githubusercontent.com/27751735/63810189-d1f8d380-c92c-11e9-8e10-db95fbed9066.png)
![image](https://user-images.githubusercontent.com/27751735/63810195-d624f100-c92c-11e9-99a7-85c6bee0864e.png)
