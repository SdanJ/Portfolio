#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:25:50 2020

@author: jinshengdan
"""
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
import re
import os

from sklearn.ensemble import RandomForestClassifier
import string
import numpy as np



##############################################################



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
#


FinalDF=pd.DataFrame()
FinalDF_TFIDF=pd.DataFrame()                    




## Add column
#print("Adding new column....")

for name in ["独り身", "一人暮らし"]:

    builder=name+"DF"
    #print(builder)
    builderB=name+"DFB"
    path="/Users/jinshengdan/Desktop/Tweets/"+name
    
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



# Model

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



for i in [1,2]:
    temp1=str("TrainDF"+str(i))
    temp2=str("Train"+str(i)+"Labels")
    temp3=str("TestDF"+str(i))
    temp4=str("Test"+str(i)+"Labels")
    
    ## perform DT
    MyDT.fit(eval(temp1), eval(temp2))
    ## plot the tree
    tree.plot_tree(MyDT)
    plt.savefig('/Users/jinshengdan/Desktop/temp1')
    feature_names=eval(str(temp1+".columns"))
    dot_data = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=eval(str(temp1+".columns")),  
                      #class_names=MyDT.class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)                                    
    graph = graphviz.Source(dot_data) 
    ## Create dynamic graph name
    tempname=str("Graph" + str(i))
    graph.render(tempname) 
    ## Show the predictions from the DT on the test set
    print("\nActual for DataFrame: ", i, "\n")
    print(eval(temp2))
    print("Prediction\n")
    DT_pred=MyDT.predict(eval(temp3))
    print(DT_pred)
    ## Show the confusion matrix
    bn_matrix = confusion_matrix(eval(temp4), DT_pred)
    print("\nThe confusion matrix is:")
    print(bn_matrix)
    FeatureImp=MyDT.feature_importances_   
    indices = np.argsort(FeatureImp)[::-1]
    ## print out the important features.....
    for f in range(TrainDF1.shape[1]):
        if FeatureImp[indices[f]] > 0:
            print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
            print ("feature name: ", feature_names[indices[f]])



# Random Forest for Text Data

RF = RandomForestClassifier()
RF.fit(TrainDF2, Train2Labels)
RF_pred=RF.predict(TestDF2)

bn_matrix_RF_text = confusion_matrix(Test2Labels, RF_pred)
print("\nThe confusion matrix is:")
print(bn_matrix_RF_text)

# VIS RF
## Feature names
FeaturesT=TrainDF2.columns


figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

tree.plot_tree(RF.estimators_[0],
               feature_names = FeaturesT, 
               filled = True)

##save it
figT.savefig('RF_Tree_Text')  ## creates png

# View estimator Trees in RF

figT2, axesT2 = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)

for index in range(0, 5):
    tree.plot_tree(RF.estimators_[index],
                   feature_names = FeaturesT, 
                   filled = True,
                   ax = axesT2[index])

    axesT2[index].set_title('Estimator: ' + str(index), fontsize = 11)
## Save it
figT2.savefig('FIVEtrees_RF.png')


## Feature importance in RF

## Recall that FeaturesT are the columns names - the words in this case.
FeatureImpRF=RF.feature_importances_   
indicesRF = np.argsort(FeatureImpRF)[::-1]
## print out the important features.....
for f2 in range(TrainDF2.shape[1]):   ##TrainDF2.shape[1] is number of columns
    if FeatureImpRF[indicesRF[f2]] >= 0.01:
        print("%d. feature %d (%.2f)" % (f2 + 1, indicesRF[f2], FeatureImpRF[indicesRF[f2]]))
        print ("feature name: ", FeaturesT[indicesRF[f2]])
        

## PLOT THE TOP 10 FEATURES...........................


import matplotlib.pyplot as plt
# Set Japanese Characters
from matplotlib.font_manager import FontProperties
fp = matplotlib.font_manager.FontProperties(fname=os.path.expanduser('~/Library/Fonts/NotoSansJP-Regular.otf'))
# Plot
top_ten_arg = indicesRF[:10]
print(top_ten_arg)
plt.title('Feature Importances 独り身 and 一人暮らし',fontproperties=fp)
plt.barh(range(len(top_ten_arg)), FeatureImpRF[top_ten_arg], color='b', align='center')
plt.yticks(range(len(top_ten_arg)), [FeaturesT[i] for i in top_ten_arg],fontproperties=fp)
plt.xlabel('Relative Importance')
plt.show()



