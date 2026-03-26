import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
# GradientBoostingClassifier, ExtraTreesClassifier
from sklearn import svm
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# plot_confusion_matrix
# from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import seaborn as sn

# import the diabetic retinopathy dataset


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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# classifiers list

svmmodel = svm.SVC(kernel='rbf', gamma=0.0001, C=1000)
knnmodel = KNeighborsClassifier(n_neighbors=44)
abcmodel = AdaBoostClassifier(n_estimators=164, learning_rate=1)
rfmodel = RandomForestClassifier(n_estimators=1000, random_state=42, max_leaf_nodes=24, max_depth=1100)
gnbmodel = GaussianNB()
gpcmodel = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=42)

# list of classifiers
classifier_models = [svmmodel, knnmodel, abcmodel, rfmodel, gnbmodel, gpcmodel]
# list to store accuracy scores
accuracy_scores = []
classifier_names = ['SVM', 'KNN', 'ABC', 'RF', 'GNB', 'GPC']


print('\n', '\n', '\n', "The results for each Classification Algorithm", '\n', '\n', '\n')
# these methods fit training and testing data on each classifier and gives results, accuracy scores, confusion matrix


def classify(classifierlist):
    for i in range(len(classifierlist)):
        classifierlist[i].fit(X_train, y_train.ravel())
        y_predict = classifierlist[i].predict(X_test)
        accuracy = accuracy_score(y_test, y_predict)
        accuracy_scores.append(accuracy)
        confusionmatrix = confusion_matrix(y_test, y_predict)
        model_classification_report = classification_report(y_test, y_predict)
        cv_score = cross_val_score(classifierlist[i], X_test, y_test.ravel(), cv=10)
        print(classifierlist[i], "Accuracy", accuracy, '\n', "Confusion Matrix", '\n', confusionmatrix, '\n',
              "10 cross fold  validation score : {}".format(np.mean(cv_score)), '\n', model_classification_report, '\n', "-----------------------------"                                                                                                        "-----------------------------", '\n')
        sn.heatmap(confusionmatrix, annot=True)
        plt.title(classifierlist[i])
        plt.show()
    return

# plots a line graphy against the corresponding classifier


def graph(a, b):
    for x in a, b:
        plt.plot(a, b)
        plt.ylabel("accuracy")
        plt.xlabel("Classifiers")
        plt.show()
        return
# plots the accuracy bar graph for all classifiers


def bargraph(a, b):
    plt.bar(a, b, color='blue')
    plt.show()
    return

# main method, runs all the methods


def main():
    classify(classifier_models)
    graph(classifier_names, accuracy_scores)
    bargraph(classifier_names, accuracy_scores)


if __name__ == '__main__':
    main()
