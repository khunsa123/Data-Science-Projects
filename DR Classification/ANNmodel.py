import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score
# plot_confusion_matrix
import seaborn as sn
from tensorflow import keras
# from tensorflow.keras.utils import plot_model
from keras.utils.vis_utils import plot_model
# import the diabetic retinopathy dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint
import kerjapraktik as kp



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

# fix random seed for reproducibility
seed = 10
np.random.seed(seed)

# splitting the data into training and testing
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_dum, X_test, y_dum, y_test = train_test_split(X, y, test_size=0.2, stratify=np.array(y), random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=np.array(y), random_state=12)

# Functional API

# create a model
func_model = kp.build_ann()

func_model.summary()


func_model.save('func_model.h5')

checkpoint = ModelCheckpoint('func_model.h5', save_best_only=True)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

history = func_model.fit(x = X_train, y=y_train,
                        validation_data=(X_val, y_val),
                        batch_size=8, epochs=500,
                        callbacks=[checkpoint, early_stopping],
                        verbose=1)
# evaluate the model
scores = func_model.evaluate(X_train, y_train)
print('%s: %.2f%%' % (func_model.metrics_names[1], scores[1]*100))
epoch_list = list(range(1,59)) # EPOCH = 150
y_train_acc = history.history['accuracy']
y_val_acc = history.history['val_accuracy']
y_train_loss = history.history['loss']
y_val_loss = history.history['val_loss']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
t = f.suptitle('Artificial Neural Network', fontsize=12)

ax1.plot(epoch_list, y_train_acc, label='Train Accuracy')
ax1.plot(epoch_list, y_val_acc, label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 59, 2))
#ax1.set_ylim(0.2,1)
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, y_train_loss, label='Train Loss')
ax2.plot(epoch_list, y_val_loss, label='Validation Loss')
ax2.set_xticks(np.arange(0, 59, 2))
#ax2.set_ylim(0,1)
ax2.set_ylabel('Cross Entropy')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

#plt.savefig('study_case/acc_loss2.png')

func_acc, func_rec, func_prec, func_f1, func_cm = kp.test_ann_model(func_model, X_train, y_train, X_test, y_test)

print('Accuracy: %.3f ' % (func_acc*100))
print('Precision: %.3f ' % (func_rec*100))
print('Recall: %.3f ' % (func_prec*100))
print('F1 Score: %.3f ' % (func_f1*100))
sn.heatmap(func_cm, annot=True, fmt='d')
#plt.savefig('study_case/func_ann_cm.png')

plot_model(func_model, to_file='funcAPI.png', show_shapes=True)