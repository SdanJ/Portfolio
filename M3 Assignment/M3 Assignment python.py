#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:41:28 2020

@author: jinshengdan
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

############################################ Record Data #########################################################
df_record=pd.read_csv('part_JPN.csv') ##read in record data
print(df_record) ##record dataframe with label

# Make a copy of two dataset to keep label safe in original dataframe
df_record_copy=df_record ##make a copy of df_record to make the label safe in df_record

## Further changes will be implemented on the copies
df_record_copy=df_record_copy.drop(['Year'],axis=1) ##remove label for record data, axis=1 to remove the whole column


print(df_record_copy) ##record dataframe without label

## Standarize record data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df_record_copy=scaler.fit_transform(df_record_copy)


## Implement K-Means Clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# fit the data with k=2,3,4
for i in range(2,5):
    kmeans_record=KMeans(n_clusters=i).fit(scaled_df_record_copy)
    # get the cluster centroids
    centroids= kmeans_record.cluster_centers_
    # print the plot
    print("When k=",i,':')
    plt.scatter(scaled_df_record_copy[:,0], scaled_df_record_copy[:,1], c= kmeans_record.labels_)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.title('K-Means Clustering-Record Data')
    plt.show()
    

## Implement Hierarchical Clustering
import scipy.cluster.hierarchy as shc
dend=shc.dendrogram(shc.linkage(scaled_df_record_copy,method='ward'),labels=['2019','2010','2001','1992','1983'])
plt.title('Hierarchical Clustering-Record Data')


## 3D visualization
from mpl_toolkits import mplot3d
ax = plt.axes(projection="3d")
ax.scatter3D(scaled_df_record_copy[:,0], scaled_df_record_copy[:,1],c= kmeans_record.labels_)
plt.title('3D plotting - Record Data')
plt.show()


## Heatmaps
import seaborn as sns
plt.title('Heatmaps-Record Data')
sns.heatmap(scaled_df_record_copy,annot=True)







############################################### Text Data #######################################################
text=['tw_text1.txt','tw_text2.txt','tw_text3.txt','tw_text4.txt','tw_text5.txt']

vectorizer=CountVectorizer(input='filename',max_features=80)
vector=vectorizer.fit_transform(text)
col=vectorizer.get_feature_names() #get columns' names
df_text=pd.DataFrame(vector.toarray(),columns=col,index=text) #change index into each file's name
print(df_text) ##text dataframe with label

# Make a copy of two dataset to keep label safe in original dataframe
df_text_copy=df_text ##make a copy of df_text to make the label safe in df_text


## Further changes will be implemented on the copies
df_text_copy=df_text_copy.drop(df_text_copy.columns[0],axis=1) ##remove label for text data, axis=1 to remove the whole column
print(df_text_copy) ##text dataframe without label



##Implement K-Means Clustering
# fit the data with k=2,3,4
for i in range(2,5):
    kmeans_text=KMeans(n_clusters=i).fit(df_text_copy)
    # get the cluster centroids
    centroids= kmeans_text.cluster_centers_
    # print the plot
    print("When k=",i,':')
    plt.scatter(df_text_copy.iloc[:,0], df_text_copy.iloc[:,1], c= kmeans_text.labels_)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.title('K-Means Clustering-Text Data')
    plt.show()


## Implement Hierarchical Clustering
import scipy.cluster.hierarchy as shc
dend=shc.dendrogram(shc.linkage(df_text_copy,method='ward'),labels=['text1','text2','text3','text4','text5'])
plt.title('Hierarchical Clustering-Text Data')


## 3D visualization
ax = plt.axes(projection="3d")
ax.scatter3D(df_text_copy.iloc[:,0], df_text_copy.iloc[:,1],c= kmeans_text.labels_)
plt.title('3D plotting - Text Data')
plt.show()


## Wordclouds
# for text1
from wordcloud import WordCloud
    
textdata1=open('tw_text1.txt','r').read()
wordcloud=WordCloud(width=500,height=600,background_color='white',min_font_size=12).generate(textdata1)
plt.figure(figsize = (4, 4), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Wordcloud for text 1')  
plt.show()


textdata2=open('tw_text2.txt','r').read()
wordcloud=WordCloud(width=500,height=600,background_color='white',min_font_size=12).generate(textdata2)
plt.figure(figsize = (4, 4), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)   
plt.title('Wordcloud for text 2')
plt.show()

textdata3=open('tw_text3.txt','r').read()
wordcloud=WordCloud(width=500,height=600,background_color='white',min_font_size=12).generate(textdata3)
plt.figure(figsize = (4, 4), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)   
plt.title('Wordcloud for text3')
plt.show()

textdata4=open('tw_text4.txt','r').read()
wordcloud=WordCloud(width=500,height=600,background_color='white',min_font_size=12).generate(textdata4)
plt.figure(figsize = (4, 4), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)   
plt.title('Wordcloud for text4')
plt.show()

textdata5=open('tw_text5.txt','r').read()
wordcloud=WordCloud(width=500,height=600,background_color='white',min_font_size=12).generate(textdata5)
plt.figure(figsize = (4, 4), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)   
plt.title('Wordcloud for text5')
plt.show()


## Heatmaps
plt.title('Heatmaps-Text Data')
sns.heatmap(df_text_copy)
