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

#K-Means Clustering
X=crime_data.values[:,3:]
print(X[:10])

clusterNum=3
k_means=KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

crime_data["Clus_km"] = labels
print(crime_data.head(5))
print(crime_data.groupby('Clus_km').mean())

#Visualising the clusters
#Distribution of districts based on Murder and Total IPC Crimes
plt.scatter(X[:, 0], X[:, 9],c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Murder', fontsize=18)
plt.ylabel('Total IPC Crimes', fontsize=16)
print(plt.show())

#Distribution of districts based on Rape and Total IPC Crimes
plt.scatter(X[:, 1], X[:, 9],c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Rape', fontsize=18)
plt.ylabel('Total IPC Crimes', fontsize=16)
print(plt.show())

#Distribution of districts based on Sexual Harassment and Total IPC Crimes
plt.scatter(X[:, 6], X[:, 9],c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Sexual Harassment', fontsize=18)
plt.ylabel('Total IPC Crimes', fontsize=16)
print(plt.show())
