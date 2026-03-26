#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import seaborn as sns


#import the both pollution datasets
from google.colab import drive
drive.mount('/content/gdrive')

dataset = pd.read_csv('/content/gdrive/MyDrive/pollutionData190100.csv')
dataset2 = pd.read_csv('/content/gdrive/MyDrive/pollutionData209960.csv')

#Dataframe Width
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

#Print all column names in the dataset
print(dataset.head())

#Print all column names in dataset2
print(dataset2.head())

print(dataset.describe())

print(dataset2.describe())

dataset

dataset2

#Check correlation of attributes in both datasets
sn.heatmap(dataset.corr(), annot=True)
plt.show()

sn.heatmap(dataset2.corr(), annot=True)
plt.show()

#Print correlation in numbers
print(dataset.corr())

print(dataset2.corr())

#Check dimensions of both datasets
print("Pollution dataset dimensions : {} ".format(dataset.shape))

print("Pollution dataset2 dimensions : {} ".format(dataset2.shape))

#check shape of both datasets
print(dataset.shape)

print(dataset2.shape)

#check for missing values
print(dataset.isnull().sum().sum())

print(dataset2.isnull().sum().sum())

#Count values for each attribute
print(dataset['ozone'].value_counts())

print(dataset2['ozone'].value_counts())

#Split both datasets
print(dataset.columns)

print(dataset2.columns)

#plot attributes
sn.lineplot(data=dataset, x="ozone", y="timestamp")

sn.lineplot(data=dataset2, x="ozone", y="timestamp")

#Pair plot
sn.pairplot(dataset)

#Experiments with dataset
#creating a copy of dataset
dfcopy = dataset.copy()
dfcopy = dfcopy.drop(columns=['longitude', 'latitude','timestamp'])
# print(dfcopy.describe())
print(dfcopy.head())

dfcopy.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

dfcopy.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

import pandas
from pandas.plotting import scatter_matrix
scatter_matrix(dfcopy)
plt.show()

dfcopy.hist()
plt.show()

#top_medians = dfcopy[dfcopy["timestamp"] > 60000].sort_values("timestamp")
#top_medians.plot(x="timestamp", y=["ozone", "carbon_monoxide", "sulfure_dioxide", "nitrogen_dioxid"], kind="bar")
#dfcopy.plot(x="timestamp", y=["ozone", "carbon_monoxide", "sulfure_dioxide", "nitrogen_dioxid"], kind="bar")
dfcopy.plot(x="ozone", y="carbon_monoxide", kind="scatter")

