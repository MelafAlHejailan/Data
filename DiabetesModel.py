import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , f1_score ,recall_score , precision_score,accuracy_score ,confusion_matrix ,roc_curve, auc, roc_auc_score
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
dataset2 = pd.read_csv("/Users/reem/Desktop/Datasets/export_diabetes2.csv")
#Split dataset
KNN= KNeighborsClassifier()
#print("dimension of Hypertension data: {}".format(dataset2.shape))


feature_names =['Glucose','BloodPressure', 'SkinThickness', 'BMI','DiabetesPedigreeFunction','Age']
X=dataset2[feature_names]
y= dataset2['Outcome']
#print(X_rfe.head())
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=30,random_state=1)

model= KNN.fit(X_train,y_train)

predict=model.predict(X_test)
#model_path = '\Desktop\MachineLearning'
#pickle.dump(model, open('diabetes_prediction2.pkl', 'wb'))

#saving the model
model_path = '\Desktop\Models'
#pickle.dump(model, open(model_path + 'finalized_model.sav', 'wb'))
pickle.dump(model, open(model_path + 'diabetes_prediction2.pkl','wb'))