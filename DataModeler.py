# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:30:24 2017
@author: amitk6925@gmail.com
"""
import json
import pickle
from time import sleep
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

TRAINFILENAME = "creditcard.csv"
TEST_FR = 0.1

'#discrete features column name'
DISC_FEATURES_COL_TO_USE = []

'#continuous features column name'
CONT_FEATURES_COL_TO_USE=["Amount","V1","V2","V3","V4","V5","V6","V7","V8","V9",
"V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22",
"V23","V24","V25","V26","V27","V28"]

'#target feature column name(discrete)'
DISC_TARGET_COL_TO_USE = "Class"

'#model dump file name'
MODEL_FILENAME = "finalized_model.pkl"

def dataCleaner(dataframe):
  """
  Removes the empty rows
  :arg: DataFrame dataframe
  :return: DataFrame dataframe
  """
  dataframe = dataframe.dropna(how='all')
  for col in dataframe:
    dataframe[col] = dataframe[col].apply(lambda x : np.nan() if str(x).isspace() else x)
    dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
  return dataframe


def scalarNormalizer(df):
  """
  Normalizes the scalars using the following formula
  x_updated=(x-x_mean)/sigma^2
  :arg: DataFrame
  :return: DataFrame
  """
  arr=dict()
  for col in CONT_FEATURES_COL_TO_USE:
    mean, std =df[col].mean(), df[col].std()
    df[col]=df[col].apply(lambda x: (x-mean)/std)
    arr[col] = [mean, std]
  json.dump(arr, open('normalize.json', 'w'))
  return df


def trainEncode(df):
  '''
  encodes the discrete features into the numerical labels and stores the labels for future reference
  in a json file
  :arg: DataFrame df
  :return: DataFrame df
  '''
  # df_coded=pd.DataFrame(columns=list(set(df.columns)-set(CONT_FEATURES_COL_TO_USE)))
  _dict=dict()
  for col in DISC_FEATURES_COL_TO_USE:
    fact=pd.factorize(df[col])
    df[col]=fact[0]
    arr=[]
    for x in fact[1].tolist():
      if isinstance(x, np.int64): arr.append(int(x))
      else: arr.append(x)
    _dict[col]=arr
  json.dump(_dict, open('labels.json', 'w'))
  return df


def predictionEncode(df):
  '''
  encodes the prediction DataFrame into numerical format
  :param DataFrame df:
  :return: encoded DataFrame df
  '''
  lables=json.load(open('labels.json', 'r'))
  for col in DISC_FEATURES_COL_TO_USE:
    df[col]=df[col].apply(lambda x: lables[col].index(x))

  scale=  json.load(open('normalize.json', 'r'))
  for col in CONT_FEATURES_COL_TO_USE:
    mean, std= scale[col][0], scale[col][1]
    df[col]=df[col].apply(lambda x: (x-mean)/std)
  return df


def visualizeHistogram(df):
  '''
  Plot Histogram of classes
  :arg: dataframe
  '''
  count_classes = pd.value_counts(df[DISC_TARGET_COL_TO_USE], sort=True).sort_index()
  count_classes.plot(kind='bar')
  plt.title("Fraud class histogram")
  plt.xlabel("Class")
  plt.ylabel("Frequency")
  print("Please close the plot windows to proceed futher!")
  plt.show()


def visualizePCA(df):
  '''
  Do PCA analysis and show into a graph!
  :arg: DataFrame dataframe
  '''
  print("Doing PCA")
  pca = PCA(n_components=2)
  proj = pca.fit_transform(df[DISC_FEATURES_COL_TO_USE+CONT_FEATURES_COL_TO_USE])
  plt.scatter(proj[:, 0], proj[:, 1], c=df[DISC_TARGET_COL_TO_USE])
  plt.colorbar()
  plt.title("PCA Diagram")
  plt.xlabel("Component1")
  plt.ylabel("Component2")
  print("Please close the plot windows to proceed futher!")
  plt.show(block=False)


def showConfusionMatrix(cm):
  '''
  Shows the confusion matrix
  :arg: confusion matrix object
  '''
  plt.matshow(cm)
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


def resampling(X,y):
  print('Minority Class percentage:%0.4f'%(100*len(y[y==1])/len(y)))
  return SMOTE(random_state=44).fit_sample(X, y)


def plotROC(y,result):
  metrics.roc_auc_score(y, result)


def training(df, type=None):
  """
  Trains the model and dumps it into a pickle file
  :arg: DataFrame dataframe
  :arg: type of model to be used from:
      LR: Logistic Regression
      SVM: Support Vector Classifier
      RF: Random forest
      GBC: Gradient Boosting Classifier
      Default: Naive Bayes
  """
  df=dataCleaner(df[DISC_FEATURES_COL_TO_USE+CONT_FEATURES_COL_TO_USE+[DISC_TARGET_COL_TO_USE]])
  print("Using %d numbers of features"%len(DISC_FEATURES_COL_TO_USE + CONT_FEATURES_COL_TO_USE))
  df_coded = trainEncode(df)
  df_coded = scalarNormalizer(df_coded)
  visualizeHistogram(df_coded)
  # visualizePCA(df_coded)
  df_shuffled = df_coded.sample(frac=1, random_state=100).reset_index(drop=True)
  X, y = df_shuffled[DISC_FEATURES_COL_TO_USE + CONT_FEATURES_COL_TO_USE], df_shuffled[DISC_TARGET_COL_TO_USE]
  X, y = resampling(X, y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_FR, random_state = 42)

  print("Training the classifier!")
  if type=='LR':
    print("Using Logistic Regression Classifier")
    cls=LogisticRegression(n_jobs=2, class_weight='balanced', tol=1e-4, C=1,random_state=111)
  elif type=='SVM':
    print("Using Support Vector Machine Classifier")
    cls=SVC(class_weight='balanced', probability=True)
  elif type=='RF':
    print("Using Random Forst Classifier")
    cls=RandomForestClassifier( n_jobs=3, n_estimators=8192, class_weight='balanced', max_depth=8,
                             min_samples_leaf=1, random_state=24)
  elif type=='GBC':
    print("Using Gradient Boosting Classifier")
    cls = GradientBoostingClassifier(n_estimators=2048, max_depth=4,
                                   subsample=0.8, learning_rate=0.004,
                                   random_state=34, min_samples_split=4,
                                   max_features=
                                   int(0.4*len(DISC_FEATURES_COL_TO_USE+
                                               CONT_FEATURES_COL_TO_USE)))
  else:
    print("Using Naive Bayes Classifier")
    cls = GaussianNB()
  model = cls.fit(X_train, y_train)
  print ("Cross-validated scores:", cross_val_score(model, X_train, y_train, cv=10))
  print ("Score:", model.score(X_test, y_test))
  predict_test = model.predict(X_test)

  print('precision_score=%f\nrecall_score=%f'%(precision_score(y_test, predict_test),recall_score(y_test, predict_test)))

  print(metrics.roc_auc_score(y_test, predict_test))

  cm=confusion_matrix(y_test, predict_test)
  print("Confusion matrix:\n" + str(cm))
  # showConfusionMatrix(cm)

  pickle.dump(model, open(MODEL_FILENAME, 'wb'))
  print("Model Created!")


def prediction(df):
  '''Loads the model and predicts using that'''
  loaded_model = pickle.load(open(MODEL_FILENAME, 'rb'))
  df=dataCleaner(df[DISC_FEATURES_COL_TO_USE+CONT_FEATURES_COL_TO_USE])
  df_coded=predictionEncode(df)
  df['Prediction']=loaded_model.predict(df_coded)
  df.to_csv('result.csv')
  return df


def main():
  trainFrameName=pd.read_csv(TRAINFILENAME)
  predFrameName=pd.read_csv(TRAINFILENAME)
  training(trainFrameName, type='LR')
  prediction(predFrameName)


if __name__=="__main__":
  main()
