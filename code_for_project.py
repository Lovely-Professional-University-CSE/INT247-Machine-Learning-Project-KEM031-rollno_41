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
