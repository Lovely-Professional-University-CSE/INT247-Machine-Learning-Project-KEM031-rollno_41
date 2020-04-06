import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
%matplotlib inline

#reading the dataset crime_data.csv
crime_df=pd.read_csv('crime_data.csv')
print(crime_df.head())
print(crime_df.columns)

#Selecting some of the important features
crime_data=crime_df[['States/UTs','District','Year','Murder','Rape','Dacoity','Riots','Forgery','Acid attack','Sexual Harassment','Incidence of Rash Driving','HumanTrafficking','Total Cognizable IPC crimes']]
print(crime_data.head())

#Renaming one of the columns
crime_data=crime_data.rename(columns={'Total Cognizable IPC crimes':'Total IPC Crimes'})
print(crime_data.head())

#Visualing the data
#Murder committed in each States/UTs
x = crime_data.groupby('States/UTs')['Murder'].sum().sort_values()
plt.figure(figsize=(11,8))
plt.xlabel("Murders")
plt.ylabel("States/UTs")
print(x.plot(kind='barh'))

#Total IPC Crimes in each State
y = crime_data.groupby('States/UTs')['Total IPC Crimes'].sum().sort_values()
plt.figure(figsize=(11,8))
plt.xlabel("Murders")
plt.ylabel("States/UTs")
print(x.plot(kind='barh',color='y'))

#Total number of each type of Crime
crimes=['Murder','Rape','Dacoity','Riots','Sexual Harassment']
number=[crime_data[crime].sum() for crime in crimes]
plt.title("Crime data of India 2014")
plt.xlabel("Number of crimes")
plt.ylabel("Crimes")
plt.bar(crimes,number,0.5,color='grey')
print(plt.show())

#Scatterplot
fig, ax = plt.subplots()
ax.scatter(crime_data["Murder"], crime_data["Rape"],c='orange')
plt.xlabel("Murder")
plt.ylabel("Rape")
plt.title("Crime Data Scatterplot 2014")
print(plt.show())

#Histogram
fig, ax = plt.subplots()
ax.hist(crime_data['Murder'], range=(0,200), align='mid', histtype='stepfilled',color='r')
print(plt.show())


