import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


hyper = pd.read_csv('femalepre2.csv').drop('id',axis=1)

feature_names = ['Age', 'Obese', 'bmi', 'whr', 'SBP', 'DBP']
X = hyper[feature_names]
y = hyper['hyper']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30 ,random_state=1)
model= DecisionTreeClassifier().fit(X_train,y_train)
predict=model.predict(X_test)

# saving the model
model_path = '\Desktop\MachineLearning'
#pickle.dump(model, open(model_path + 'finalized_model.sav', 'wb'))
pickle.dump(model, open(model_path + 'female_prediction.pkl','wb'))


    

