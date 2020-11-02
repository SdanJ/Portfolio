#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:25:50 2020

@author: jinshengdan
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

from sklearn.ensemble import RandomForestClassifier
import string
import numpy as np


MyVect_IFIDF=TfidfVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        #binary=True
                        )
#

FinalDF_TFIDF=pd.DataFrame()





path="/Users/jinshengdan/Desktop/Tweet_Corpus"

FileList=[]
for item in os.listdir(path):
    #print(path+ "/" + item)
    next=path+ "/" + item
    FileList.append(next)  
    print("full list...")
    #print(FileList)
    

    X=MyVect_IFIDF.fit_transform(FileList)

    ColumnNames=MyVect_IFIDF.get_feature_names()
    NumFeatures=len(ColumnNames)

    

builderTS=pd.DataFrame(X.toarray(),columns=ColumnNames)


## Add column
#print("Adding new column....")

builderTS["Label"]='一人暮らし'

print(builderTS)


FinalDF_TFIDF= FinalDF_TFIDF.append(builderTS)
   
print(FinalDF_TFIDF.head())


## Replace the NaN with 0 because it actually 
## means none in this case
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
FinalDF_TFIDF=RemoveNums(FinalDF_TFIDF)

## Have a look:
print(FinalDF_TFIDF)



# Model

## Create the testing set - grab a sample from the training set. 

from sklearn.model_selection import train_test_split
import random as rd
rd.seed(1234)
TrainDF, TestDF = train_test_split(FinalDF_TFIDF, test_size=0.3)




# For all three DFs - separate LABELS
## Save labels
### TEST 
TestLabels=TestDF["Label"]
print(TestLabels)
## remove labels
TestDF = TestDF.drop(["Label"], axis=1)
print(TestDF)

## TRAIN 
TrainLabels=TrainDF["Label"]
print(TrainLabels)
## remove labels
TrainDF = TrainDF.drop(["Label"], axis=1)
print(TrainDF)


# Decision Trees 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix


MyDT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)
# Perfoem DT
MyDT.fit(TrainDF, TrainLabels)

## plot the tree
tree.plot_tree(MyDT)
plt.savefig("/Users/jinshengdan/Desktop/TrainDF")
feature_names=TrainDF.columns
dot_data = tree.export_graphviz(MyDT, out_file=None,
                ## The following creates TrainDF.columns for each
                ## which are the feature names.
                  feature_names=TrainDF.columns,  
                  #class_names=MyDT.class_names,  
                  filled=True, rounded=True,  
                  special_characters=True)                                    
graph = graphviz.Source(dot_data) 
## Create dynamic graph name
tempname="Graph" 
graph.render(tempname) 
## Show the predictions from the DT on the test set
print("\nActual for DataFrame: \n")
print(TrainLabels)
print("Prediction\n")
DT_pred=MyDT.predict(TestDF)
print(DT_pred)
## Show the confusion matrix
bn_matrix = confusion_matrix(TestLabels, DT_pred)
print("\nThe confusion matrix is:")
print(bn_matrix)
FeatureImp=MyDT.feature_importances_   
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])


# Random Forest for Text Data

RF = RandomForestClassifier()
RF.fit(TrainDF, TrainLabels)
RF_pred=RF.predict(TestDF)

bn_matrix_RF_text = confusion_matrix(TestLabels, RF_pred)
print("\nThe confusion matrix is:")
print(bn_matrix_RF_text)

# VIS RF
## Feature names
FeaturesT=TrainDF.columns


figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

tree.plot_tree(RF.estimators_[0],
               feature_names = FeaturesT, 
               filled = True)

##save it
figT.savefig('RF_Tree_Text_')  ## creates png

# View estimator Trees in RF

figT2, axesT2 = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)

for index in range(0, 5):
    tree.plot_tree(RF.estimators_[index],
                   feature_names = FeaturesT, 
                   filled = True,
                   ax = axesT2[index])

    axesT2[index].set_title('Estimator: ' + str(index), fontsize = 11)
## Save it
figT2.savefig('FIVEtrees_RF_.png')


## Feature importance in RF

## Recall that FeaturesT are the columns names - the words in this case.
FeatureImpRF=RF.feature_importances_   
indicesRF = np.argsort(FeatureImpRF)[::-1]
## print out the important features.....
for f2 in range(TrainDF.shape[1]):   ##TrainDF1.shape[1] is number of columns
    if FeatureImpRF[indicesRF[f2]] >= 0.01:
        print("%d. feature %d (%.2f)" % (f2 + 1, indicesRF[f2], FeatureImpRF[indicesRF[f2]]))
        print ("feature name: ", FeaturesT[indicesRF[f2]])
        

## PLOT THE TOP 10 FEATURES...........................
top_ten_arg = indicesRF[:10]
print(top_ten_arg)
plt.title('Feature Importances')
plt.barh(range(len(top_ten_arg)), FeatureImpRF[top_ten_arg], color='b', align='center')
plt.yticks(range(len(top_ten_arg)), [FeaturesT[i] for i in top_ten_arg])
plt.xlabel('Importance')
plt.show()
