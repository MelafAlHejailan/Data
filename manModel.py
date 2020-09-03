import numpy as np 
import pandas as pd 
import re
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , f1_score ,recall_score , precision_score,accuracy_score ,confusion_matrix ,roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
#import mlxtend
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import pca
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pickle

hyper = pd.read_csv('manpre2.csv')
feature_names = ['Age', 'Obese', 'bmi', 'whr', 'SBP', 'DBP']
X = hyper[feature_names]
y = hyper['hyper']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30 ,random_state=1)
model= DecisionTreeClassifier().fit(X_train,y_train)
predict=model.predict(X_test)

# saving the model
model_path = '\Desktop\MachineLearning'
#pickle.dump(model, open(model_path + 'finalized_model.sav', 'wb'))
pickle.dump(model, open(model_path + 'man_prediction.pkl','wb'))
