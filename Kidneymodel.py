import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


Kidney = pd.read_csv("export_CKD5-7-11AllFeatures.csv")
#Split dataset
print("dimension of Hypertension data: {}".format(Kidney.shape))



feature_names =['age', 'bp', 'sg', 'al', 'su', 'rbc', 'ba', 'bu', 'pot', 'wc', 'cad',
       'appet', 'pe']
X=Kidney[feature_names ]
y= Kidney['classification']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y,test_size=30,random_state=1)
print(X.head())
model= DecisionTreeClassifier().fit(X_train1,y_train1)
predict=model.predict(X_test1)
#saving the model
model_path = '\Desktop\MachineLearning'
#pickle.dump(model, open(model_path + 'finalized_model.sav', 'wb'))
pickle.dump(model, open(model_path + 'kidney_prediction2.pkl','wb'))



    

