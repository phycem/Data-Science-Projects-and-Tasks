#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
sns.set()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None)
from scipy import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from scipy.stats import skew
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, plot_roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.preprocessing import RobustScaler
import pickle

import plotly.graph_objects as go
import tensorflow as tf

import plotly.express as px
from tabulate import tabulate


def read_filename(filename):
    df = pd.read_csv(filename)
    df = df.dropna(axis=0)
    return(df)
 
# dataframe with 303 rows


def process_data(df):
    
    #code that removes the columns that are not highly correlated
   
    df.drop(["chol", "fbs"], axis = 1, inplace = True)
    
    #put your code that finds outliers for trtbps and changes them 
    z_scores_trtbps = zscore(df["trtbps"])
    winsorize_percentile_trtbps = (stats.percentileofscore(df["trtbps"], 165)) / 100
    trtbps_winsorize = winsorize(df.trtbps, (0, (1 - winsorize_percentile_trtbps)))
    df["trtbps_winsorize"] = trtbps_winsorize
    df.drop(["chol", "fbs"], axis = 1, inplace = True)
    
    #code that does one hot encoding
    categoric_var=['sex', 'cp', 'restecg', 'exng', 'slp', 'caa', 'thall', 'output']
    df = pd.get_dummies(df, columns = categoric_var[:-1], drop_first = True)
    
    #robust scalar
    new_numeric_var = ["age", "thalachh", "trtbps", "oldpeak"]
    final_dataset = df.copy()
    robust_scaler = RobustScaler()
    final_dataset[new_numeric_var] = robust_scaler.fit_transform(df[new_numeric_var])
    return (final_dataset)


def model_training(final_dataset):
    X = df.drop(["output"], axis = 1)
    y = df[["output"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print ('model saved as "finalized_model.sav" in current working directory')
    print (classification_report)

   
def load_model(model_filename):
    loaded_model = pickle.load(open(model_filename, 'rb'))
    return (loaded_model)


def generate_predictions(loaded_model, final_dataset):
    y_predict = loaded_model.predict(X_test, y_test)
    print(accuracy_score(y_test,y_predict))
    return(y_predict)
   


def main():
    filename = 'demo.csv'
    model_filename = 'finalized_model.sav'
    df = read_filename(filename)
    final_dataset = process_data(df)
    model_training(final_dataset)
    loaded_model = load_model(model_filename)
    y_predict = generate_predictions(loaded_model, final_dataset)
    y_predict.to_csv('y_predict.csv')
    print('predicted values saved to file "y_predict.csv" in current working directory')
   
if __name__ =="__main__":
    main()
    

