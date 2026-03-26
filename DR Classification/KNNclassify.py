import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('DR.csv')

# pandas dataframe width
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# print all column names in the dataset
print(dataset.head())

print(dataset.describe())

# print the number of rows and columns present in the dataset
print("DR dataset dimensions : {} ".format(dataset.shape))

# shape of the data set
print(dataset.shape)
# check for missing values
print(dataset.isnull().sum().sum())

# bad quality values from dataset
print(dataset['Quality'].value_counts())

# splitting the data into dependent and independent variables
print(dataset.columns)

# independent variable X
X = dataset[['Quality',
             # 0 The binary result of quality assessment, 0 = bad quality 1 = sufficient quality
            'Abnormality',
             # 1 The binary result of pre-screening, 1 indicates severe retinal abnormality and 0 its lack
             'MA',                 # 2 Number of MAs found at the confidence levels alpha = 0.5
             'MA.1',               # 3 Number of MAs found at the confidence levels alpha = 0.6
             'MA.2',               # 4 Number of MAs found at the confidence levels alpha = 0.7
             'MA.3',               # 5 Number of MAs found at the confidence levels alpha = 0.8
             'MA.4',               # 6 Number of MAs found at the confidence levels alpha = 0.9
             'MA.5',               # 7 Number of MAs found at the confidence levels alpha = 1.0
             'Exudates',           # 8 Number of Exudates found at the confidence levels alpha = 0.5
             'Exudates.1',         # 9 Number of Exudates found at the confidence levels alpha = 0.6
             'Exudates.2',         # 10 Number of Exudates found at the confidence levels alpha = 0.7
             'Exudates.3',         # 11 Number of Exudates found at the confidence levels alpha = 0.8
             'Exudates.4',         # 12 Number of Exudates found at the confidence levels alpha = 0.9
             'Exudates.5',         # 13 Number of Exudates found at the confidence levels alpha = 1.0
             'Exudates.6',         # 14 Number of Exudates found at the confidence levels alpha = 1.0
             'Exudates.7',         # 15 Number of Exudates found at the confidence levels alpha = 1.0
             'Euclidean distance',
             # 16 The euclidean distance of the center of the macula and the center of the optic disc
             'Dia OD',        # 17 The diameter of the optic disc
             'AM/FM'               # 18 The binary result of the AM/FM-based classification
             ]]
print("The list of independent variables", '\n', X)


# dependent variable y
y = dataset[['Class']]            # classification result 1,2,3 = signs of DR, 0 = no signs of DR
print("The dependent variable ", '\n', y)


# splitting the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Calculating error for K values between 1 and 40
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# plot the error values against K values

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

