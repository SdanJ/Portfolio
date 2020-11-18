#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:29:42 2020

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
#####       Record Data - iris    #####
#######################################

from vega_datasets import data
# Load in data - iris
df = data.iris()
df.head()


# Seperate dataset into TRAINING and TESTING sets
TrainDF, TestDF= train_test_split(df, test_size=0.3)

# Seperate LABELS FROM DATA
## TEST
TestLabels=TestDF["species"]  ## save labels
print(TestLabels)
TestDF =TestDF.drop(["species"], axis=1)  ##drop labels
print(TestDF)

## TRAIN 
TrainLabels=TrainDF["species"]  ## save labels
print(TrainLabels)
TrainDF = TrainDF.drop(["species"], axis=1)  ##drop labels
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
ymap=ymap.replace("virginica", 1)
ymap=ymap.replace("setosa", 0)
ymap=ymap.replace("versicolor", 2)

pca = PCA(n_components=3)
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
TrainDF['species']=TrainLabels
TrainDF=TrainDF[(TrainDF['species']!='virginica')]
TrainDF=TrainDF.drop(['sepalLength','sepalWidth'], axis=1)
TrainDF.head()

# Convert categorical values to numerical target
TrainDF=TrainDF.replace('setosa', 0)
TrainDF=TrainDF.replace('versicolor', 1)
X=TrainDF.iloc[:,0:2]
y=TrainDF['species']
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


############################################
#####     Text Data - 'Cat'& 'Book'    #####
############################################

STEMMER=PorterStemmer()


# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words


import string
import numpy as np



MyVect_STEM=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )


MyVect_STEM_Bern=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        binary=True
                        )



MyVect_IFIDF=TfidfVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        lowercase = True,
                        #binary=True
                        )

MyVect_IFIDF_STEM=TfidfVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        #binary=True
                        )
#

#We will be creating new data frames - one for NB and one for Bern. 
## These are the two new and currently empty DFs
FinalDF_STEM=pd.DataFrame()
FinalDF_STEM_Bern=pd.DataFrame()
FinalDF_TFIDF=pd.DataFrame()
FinalDF_TFIDF_STEM=pd.DataFrame()


for name in ["Cat", "Book"]:

    builder=name+"DF"
    #print(builder)
    builderB=name+"DFB"
    path="/Users/jinshengdan/Desktop/Assignment6 practice/"+name
    
    FileList=[]
    for item in os.listdir(path):
        #print(path+ "/" + item)
        next=path+ "/" + item
        FileList.append(next)  
        print("full list...")
        #print(FileList)
        
        ## Do for all three
        ## MyVect_STEM  and MyVect_IFIDF and MyVect_IFIDF_STEM
        X1=MyVect_STEM.fit_transform(FileList)
        X2=MyVect_IFIDF.fit_transform(FileList)
        X3=MyVect_IFIDF_STEM.fit_transform(FileList)
        XB=MyVect_STEM_Bern.fit_transform(FileList)
        
        
        ColumnNames1=MyVect_STEM.get_feature_names()
        NumFeatures1=len(ColumnNames1)
        ColumnNames2=MyVect_IFIDF.get_feature_names()
        NumFeatures2=len(ColumnNames2)
        ColumnNames3=MyVect_IFIDF_STEM.get_feature_names()
        NumFeatures3=len(ColumnNames3)
        ColumnNamesB=MyVect_STEM_Bern.get_feature_names()
        NumFeatures4=len(ColumnNamesB)
        #print("Column names: ", ColumnNames2)
        #Create a name
        
   
    builderS=pd.DataFrame(X1.toarray(),columns=ColumnNames1)
    builderT=pd.DataFrame(X2.toarray(),columns=ColumnNames2)
    builderTS=pd.DataFrame(X3.toarray(),columns=ColumnNames3)
    builderB=pd.DataFrame(XB.toarray(),columns=ColumnNamesB)
    
    ## Add column
    #print("Adding new column....")
    builderS["Label"]=name
    builderT["Label"]=name
    builderTS["Label"]=name
    builderB["Label"]=name
    #print(builderS)
    
    FinalDF_STEM= FinalDF_STEM.append(builderS)
    FinalDF_STEM_Bern= FinalDF_STEM_Bern.append(builderB)
    FinalDF_TFIDF= FinalDF_TFIDF.append(builderT)
    FinalDF_TFIDF_STEM= FinalDF_TFIDF_STEM.append(builderTS)
   
    print(FinalDF_STEM.head())

## Replace the NaN with 0 because it actually 
## means none in this case
FinalDF_STEM=FinalDF_STEM.fillna(0)
FinalDF_STEM_Bern=FinalDF_STEM_Bern.fillna(0)
FinalDF_TFIDF=FinalDF_TFIDF.fillna(0)
FinalDF_TFIDF_STEM=FinalDF_TFIDF_STEM.fillna(0)

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
FinalDF_STEM=RemoveNums(FinalDF_STEM)
FinalDF_STEM_Bern=RemoveNums(FinalDF_STEM_Bern)
FinalDF_TFIDF=RemoveNums(FinalDF_TFIDF)
FinalDF_TFIDF_STEM=RemoveNums(FinalDF_TFIDF_STEM)

## Have a look:
print(FinalDF_STEM)
print(FinalDF_STEM_Bern)
print(FinalDF_TFIDF)
print(FinalDF_TFIDF_STEM)

## Create the testing set - grab a sample from the training set. 

from sklearn.model_selection import train_test_split
import random as rd
rd.seed(1234)
TrainDF1, TestDF1 = train_test_split(FinalDF_STEM, test_size=0.3)
TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF, test_size=0.3)
TrainDF3, TestDF3 = train_test_split(FinalDF_TFIDF_STEM, test_size=0.3)
TrainDF4, TestDF4 = train_test_split(FinalDF_STEM_Bern, test_size=0.4)

# For all three DFs - separate LABELS

## Save labels
### TEST ---------------------
Test1Labels=TestDF1["Label"]
Test2Labels=TestDF2["Label"]
Test3Labels=TestDF3["Label"]
Test4Labels=TestDF4["Label"]
print(Test2Labels)
## remove labels
TestDF1 = TestDF1.drop(["Label"], axis=1)
TestDF2 = TestDF2.drop(["Label"], axis=1)
TestDF3 = TestDF3.drop(["Label"], axis=1)
TestDF4 = TestDF4.drop(["Label"], axis=1)
print(TestDF1)

## TRAIN ----------------------------
Train1Labels=TrainDF1["Label"]
Train2Labels=TrainDF2["Label"]
Train3Labels=TrainDF3["Label"]
Train4Labels=TrainDF4["Label"]
print(Train3Labels)
## remove labels
TrainDF1 = TrainDF1.drop(["Label"], axis=1)
TrainDF2 = TrainDF2.drop(["Label"], axis=1)
TrainDF3 = TrainDF3.drop(["Label"], axis=1)
TrainDF4 = TrainDF4.drop(["Label"], axis=1)
print(TrainDF3)


## -------------------- Naive Bayes ---------------------- ##
#Create the modeler
MyModelNB= MultinomialNB()


## Run on all four Dfs
NB1=MyModelNB.fit(TrainDF1, Train1Labels)
Prediction1 = MyModelNB.predict(TestDF1)
print(np.round(MyModelNB.predict_proba(TestDF1),2))

NB2=MyModelNB.fit(TrainDF2, Train2Labels)
Prediction2 = MyModelNB.predict(TestDF2)
print(np.round(MyModelNB.predict_proba(TestDF2),2))

NB3=MyModelNB.fit(TrainDF3, Train3Labels)
Prediction3 = MyModelNB.predict(TestDF3)
print(np.round(MyModelNB.predict_proba(TestDF3),2))

NB4=MyModelNB.fit(TrainDF4, Train4Labels)
Prediction4 = MyModelNB.predict(TestDF4)
print(np.round(MyModelNB.predict_proba(TestDF4),2))



print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)

print("\nThe prediction from NB is:")
print(Prediction3)
print("\nThe actual labels are:")
print(Test3Labels)

print("\nThe prediction from NB is:")
print(Prediction4)
print("\nThe actual labels are:")
print(Test4Labels)

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

cnf_matrix3 = confusion_matrix(Test3Labels, Prediction3)
print("\nThe confusion matrix is:")
print(cnf_matrix3)

cnf_matrix4 = confusion_matrix(Test4Labels, Prediction4)
print("\nThe confusion matrix is:")
print(cnf_matrix4)


## remap labels to numbers to view
ymap=Train1Labels
ymap=ymap.replace("Cat", 1)
ymap=ymap.replace("Book", 0)
ymap

pca = PCA(n_components=2)
proj = pca.fit_transform(TrainDF1)
plt.scatter(proj[:, 0], proj[:, 1], c=ymap, cmap="Paired")
plt.colorbar()



#ymap2=Train2Labels
#ymap2=ymap2.replace("Cat", 1)
#ymap2=ymap2.replace("Book", 0)
#ymap2

#pca = PCA(n_components=2)
#proj = pca.fit_transform(TrainDF2)
#plt.scatter(proj[:, 0], proj[:, 1], c=ymap2, cmap="Paired")
#plt.colorbar()



#ymap3=Train3Labels
#ymap3=ymap3.replace("Cat", 1)
#ymap3=ymap3.replace("Book", 0)
#ymap3

#pca = PCA(n_components=2)
#proj = pca.fit_transform(TrainDF3)
#plt.scatter(proj[:, 0], proj[:, 1], c=ymap3, cmap="Paired")
#plt.colorbar()



#ymap4=Train4Labels
#ymap4=ymap4.replace("Cat", 1)
#ymap4=ymap4.replace("Book", 0)
#ymap4

#pca = PCA(n_components=2)
#proj = pca.fit_transform(TrainDF4)
#plt.scatter(proj[:, 0], proj[:, 1], c=ymap4, cmap="Paired")
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

import matplotlib.pyplot as plt 
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
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()

# Using the top 2 features from above. Let's look at the margin of the SVM
from sklearn.svm import SVC
X = np.array([TRAIN["book"], TRAIN["cat"]])
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







