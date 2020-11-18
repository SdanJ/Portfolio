#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:39:53 2020

@author: jinshengdan
"""


## Textmining Naive Bayes Example
import nltk
from sklearn import preprocessing
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D 
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
## conda install pydotplus
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from nltk.stem.porter import PorterStemmer


#######################################
####      Record Data - JPN.csv    ####
#######################################

# Read in dataset
JPN = pd.read_csv('JPN.csv')
JPN.head()

JPN.columns

# Seperate dataset into TRAINING and TESTING sets
TrainDF, TestDF= train_test_split(JPN, test_size=0.3)

# Seperate LABELS FROM DATA
## TEST
TestLabels=TestDF[" Birth Rate>=8.9"]  ## save labels
print(TestLabels)
TestDF =TestDF.drop([" Birth Rate>=8.9"], axis=1)  ##drop labels
print(TestDF)

## TRAIN 
TrainLabels=TrainDF[" Birth Rate>=8.9"]  ## save labels
print(TrainLabels)
TrainDF = TrainDF.drop([" Birth Rate>=8.9"], axis=1)  ##drop labels
print(TrainDF)

## -------------------- Naive Bayes ---------------------- ##
MyModel_NB=MultinomialNB()

MyModel_NB.fit(TrainDF, TrainLabels)
PredictionNB = MyModel_NB.predict(TestDF)
print("\nThe prediction from NB is:")
print(PredictionNB)
print("\nThe actual labels are:")
print(TestLabels)

# confusion matrix
cnf_matrix = confusion_matrix(TestLabels, PredictionNB)
print("\nThe confusion matrix is:")
print(cnf_matrix)

# prediction probabilities
print(np.round(MyModel_NB.predict_proba(TestDF),2))
MyModel_NB.get_params(deep=True)


## remap labels to numbers to view
ymap=TrainLabels

pca = PCA(n_components=2)
proj = pca.fit_transform(TrainDF)
plt.scatter(proj[:, 0], proj[:, 1], c=ymap, cmap="Paired")
plt.colorbar()


## -------------------- SVM ---------------------- ##
# SCALE ALL DATA to between 0 and 1 from sklearn import preprocessing
x = TrainDF.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
TrainDF_S = pd.DataFrame(x_scaled)

x2 = TestDF.values #returns a numpy array
min_max_scaler2 = preprocessing.MinMaxScaler()
x_scaled2 = min_max_scaler2.fit_transform(x2)
TestDF_S = pd.DataFrame(x_scaled2)
print(TestDF_S)

####---- Linear Kernel with C = 1
SVM_Model_1=LinearSVC(C=1)
SVM_Model_1.fit(TrainDF_S, TrainLabels)

print("SVM prediction:\n", SVM_Model_1.predict(TestDF_S))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model_1.predict(TestDF_S))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

####---- Radial Basis Kernel with C=0.5
SVM_Model_2=sklearn.svm.SVC(C=0.5, kernel='rbf', degree=3, gamma="auto")
SVM_Model_2.fit(TrainDF_S, TrainLabels)

print("SVM prediction:\n", SVM_Model_2.predict(TestDF_S))
print("Actual:")
print(TestLabels)

SVM_matrix_2 = confusion_matrix(TestLabels, SVM_Model_2.predict(TestDF_S))
print("\nThe confusion matrix is:")
print(SVM_matrix_2)
print("\n\n")



####---- Polynomial Kernel with C=0.1
SVM_Model_3=sklearn.svm.SVC(C=0.1, kernel='poly', degree=3, gamma="auto")
SVM_Model_3.fit(TrainDF_S, TrainLabels)

print("SVM prediction:\n", SVM_Model_3.predict(TestDF_S))
print("Actual:")
print(TestLabels)

SVM_matrix_3 = confusion_matrix(TestLabels, SVM_Model_3.predict(TestDF_S))
print("\nThe confusion matrix is:")
print(SVM_matrix_3)
print("\n\n")

# Visualization
TrainDF[' Birth Rate>=8.9']=TrainLabels
TrainDF=TrainDF.drop(['Year',' Population ages 65 and above (% of total population)',' GDP (current US$)'], axis=1)
TrainDF.head()


X=TrainDF.iloc[:,0:2]
y=TrainDF[' Birth Rate>=8.9']
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')

from sklearn.svm import SVC 
model = SVC(kernel='linear', C=1)
model.fit(X, y)

ax = plt.gca()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()



##################################################
#####     Text Data - 'LiveAlone'& 'Solo'    #####
##################################################

MyVect=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )

MyVect_IFIDF=TfidfVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        #binary=True
                        )

FinalDF=pd.DataFrame()
FinalDF_TFIDF=pd.DataFrame()    

## Add column
#print("Adding new column....")

for name in ["独り身", "一人暮らし"]:

    builder=name+"DF"
    #print(builder)
    builderB=name+"DFB"
    path="/Users/jinshengdan/Desktop/Assignment 6 Python/Tweets/"+name
    
    FileList=[]
    for item in os.listdir(path):
        #print(path+ "/" + item)
        next=path+ "/" + item
        FileList.append(next)  
        print("full list...")
        #print(FileList)
        
        ## Do for all three
        ## MyVect_STEM  and MyVect_IFIDF and MyVect_IFIDF_STEM
        X1=MyVect.fit_transform(FileList)
        X2=MyVect_IFIDF.fit_transform(FileList)
        
        
        ColumnNames1=MyVect.get_feature_names()
        NumFeatures1=len(ColumnNames1)
        ColumnNames2=MyVect_IFIDF.get_feature_names()
        NumFeatures2=len(ColumnNames2)

        
   
    builderS=pd.DataFrame(X1.toarray(),columns=ColumnNames1)
    builderT=pd.DataFrame(X2.toarray(),columns=ColumnNames2)

    
    ## Add column
    #print("Adding new column....")
    builderS["Label"]=name
    builderT["Label"]=name

    #print(builderS)
    
    FinalDF= FinalDF.append(builderS)
    FinalDF_TFIDF= FinalDF_TFIDF.append(builderT)

   
    print(FinalDF.head())

## Replace the NaN with 0 because it actually 
## means none in this case
FinalDF=FinalDF.fillna(0)
FinalDF_TFIDF=FinalDF_TFIDF.fillna(0)

# REMOVE number columns
## Remove columns with number from this one
## Create a function that removes columns that are/contain nums
def RemoveNums(SomeDF):
    #print(SomeDF)
    print("Running Remove Numbers function....\n")
    temp=SomeDF
    MyList=[]
    for col in temp.columns:
        #print(col)
        #Logical1=col.isdigit()  ## is a num
        Logical2=str.isalpha(col) ## this checks for anything
        ## that is not a letter
        if(Logical2==False):# or Logical2==True):
            #print(col)
            MyList.append(str(col))
            #print(MyList)       
    temp.drop(MyList, axis=1, inplace=True)
            #print(temp)
            #return temp
       
    return temp

## Call the function
FinalDF=RemoveNums(FinalDF)
FinalDF_TFIDF=RemoveNums(FinalDF_TFIDF)

## Have a look:
print(FinalDF)
print(FinalDF_TFIDF)


## Create the testing set - grab a sample from the training set. 

from sklearn.model_selection import train_test_split
import random as rd
rd.seed(1234)

TrainDF1, TestDF1 = train_test_split(FinalDF, test_size=0.3)
TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF, test_size=0.3)

# For both DFs - separate LABELS

## Save labels
### TEST ---------------------
Test1Labels=TestDF1["Label"]
Test2Labels=TestDF2["Label"]

print(Test1Labels)
print(Test2Labels)
## remove labels
TestDF1 = TestDF1.drop(["Label"], axis=1)
TestDF2 = TestDF2.drop(["Label"], axis=1)

print(TestDF1)
print(TestDF2)

## TRAIN ----------------------------
Train1Labels=TrainDF1["Label"]
Train2Labels=TrainDF2["Label"]

print(Train1Labels)
print(Train2Labels)
## remove labels
TrainDF1 = TrainDF1.drop(["Label"], axis=1)
TrainDF2 = TrainDF2.drop(["Label"], axis=1)

print(TrainDF1)
print(TrainDF2)


## -------------------- Naive Bayes ---------------------- ##
#Create the modeler
MyModelNB= MultinomialNB()

## Run on both two Dfs
NB1=MyModelNB.fit(TrainDF1, Train1Labels)
Prediction1 = MyModelNB.predict(TestDF1)
print(np.round(MyModelNB.predict_proba(TestDF1),2))

NB2=MyModelNB.fit(TrainDF2, Train2Labels)
Prediction2 = MyModelNB.predict(TestDF2)
print(np.round(MyModelNB.predict_proba(TestDF2),2))

print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)

## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
cnf_matrix1 = confusion_matrix(Test1Labels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)

cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
print("\nThe confusion matrix is:")
print(cnf_matrix2)

## remap labels to numbers to view

ymap=Train1Labels
ymap=ymap.replace("一人暮らし", 1)
ymap=ymap.replace("独り身", 0)
ymap

pca = PCA(n_components=2)
proj = pca.fit_transform(TrainDF1)
plt.scatter(proj[:, 0], proj[:, 1], c=ymap, cmap="Paired")
plt.colorbar()


#ymap2=Train2Labels
#ymap2=ymap2.replace("一人暮らし", 1)
#ymap2=ymap2.replace("独り身", 0)
#ymap2

#pca = PCA(n_components=2)
#proj = pca.fit_transform(TrainDF2)
#plt.scatter(proj[:, 0], proj[:, 1], c=ymap2, cmap="Paired")
#plt.colorbar()

## -------------------- SVM ---------------------- ##

####---- Linear Kernel with C = 1

TRAIN= TrainDF1   ## As noted above - this can also be TrainDF2, etc.
TRAIN_Labels= Train1Labels
TEST= TestDF1
TEST_Labels= Test1Labels

SVM_Model1=LinearSVC(C=1)
SVM_Model1.fit(TRAIN, TRAIN_Labels)

#print("SVM prediction:\n", SVM_Model1.predict(TEST))
#print("Actual:")
#print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

####---- Radial Basis Kernel with C= 0.5
SVM_Model2=sklearn.svm.SVC(C=0.5, kernel='rbf', 
                           verbose=True, gamma="auto")
SVM_Model2.fit(TRAIN, TRAIN_Labels)

#print("SVM prediction:\n", SVM_Model2.predict(TEST))
#print("Actual:")
#print(TEST_Labels)

SVM_matrix2 = confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix2)
print("\n\n")

####---- Polynomial Kernel with C=0.1
SVM_Model3=sklearn.svm.SVC(C=0.1, kernel='poly',degree=2,
                           gamma="auto", verbose=True)

print(SVM_Model3)
SVM_Model3.fit(TRAIN, TRAIN_Labels)

#print("SVM prediction:\n", SVM_Model3.predict(TEST))
#print("Actual:")
#print(TEST_Labels)

SVM_matrix3 = confusion_matrix(TEST_Labels, SVM_Model3.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix3)
print("\n\n")

# Visualizing the top features. Then Visualizing the margin with the top 2 in 2D
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
fp = matplotlib.font_manager.FontProperties(fname=os.path.expanduser('~/Library/Fonts/NotoSansJP-Regular.otf'))
## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
## Define a function to visualize the TOP words (variables)
def plot_coefficients(MODEL=SVM_Model1, COLNAMES=TrainDF1.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    print(top_positive_coefficients)
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    print(top_negative_coefficients)
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    # Set Japanese Characters
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right",fontproperties=fp)
    plt.show()
    

plot_coefficients()

# Using the top 2 features from above. Let's look at the margin of the SVM
from sklearn.svm import SVC
X = np.array([TRAIN["独り身"], TRAIN["一人暮らし"]])
X = X.transpose()
print(X)
#The classes of the training data
y = TRAIN_Labels
print(y)
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y=lb.fit_transform(y)

y = np.array(y)
y = y.ravel()  ## to make it the right 1D array type

print(y)


## Here - we need to make y into 0 or 1 so it will plot

#TRAIN
#Define the model with SVC
# Fit SVM with training data
clf = SVC(C=1, kernel="linear")
clf.fit(X, y) 

margin = 2 / np.sqrt(np.sum(clf.coef_ ** 2))

# get the separating hyperplane
#The weights vector w
w = clf.coef_[0]
#print("The weight vector ", w)

#The slope of the SVM sep line
a = -w[0] / w[1]
#print("The slope of the SVM sep line is ", a)

#Create a variable xx that are values between 4 and 8
xx = np.linspace(0, 10)

#Equation of sep line in 2D
# x1  = - b/w1  - (w0/w1 )(x0)
## Note that clf_intercept_[0] is "b"
## Note that a  = -w0/w1 and xx are a bunch of x values
## This is the y values for the main sep line
yy = a * xx - (clf.intercept_[0]) / w[1]

##These plot the two parellel margin lines
# plot the parallel lines to the separating hyperplane 
#that pass through the support vectors and note the margin
#margin = 2 / np.sqrt(np.sum(clf.coef_ ** 2))
#translate the location of the center sep line by
# adding or subtracting a fraaction of the margin 
yy_down = yy + .5*margin
yy_up = yy - .5*margin


# plot the line, the points, and the nearest vectors to the plane
#plt.figure(fignum, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, 'r-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
                facecolors='none', zorder=5)
#cmap is the color map
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=5, cmap=plt.cm.Paired)

plt.axis('tight')



















