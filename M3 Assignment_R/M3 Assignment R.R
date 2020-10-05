library(stats) 
install.packages("NbClust")
library(NbClust)
library(cluster)
library(mclust)
install.packages('amap')
library(amap)
library(factoextra) 
library(purrr)
install.packages("stylo")
library(stylo)  
install.packages("philentropy")
library(philentropy)
install.packages('dbscan')
library(dbscan)
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm)

############################################# Record Data ####################################################
Record_df<-read.csv('part_JPN.csv') # read in record data

# Save the label
(Label_Record_df<-Record_df$Year)

Record_df_cp<-Record_df # make a copy and further changes will be made on the copy
head(Record_df_cp)

# Remove the label
Record_df_cp<-Record_df_cp[,-c(1)]
head(Record_df_cp)

# Normalize Data
Record_df_cp_Norm<-as.data.frame(apply(Record_df_cp[,1:4], 2, ##2 for col
                                       function(x) (x - min(x))/(max(x)-min(x))))

# Distance Matrics
(Dist1<- dist(Record_df_cp_Norm, method = "minkowski", p=1)) ##Manhattan
(Dist2<- dist(Record_df_cp_Norm, method = "minkowski", p=2)) #Euclidean
(DistE<- dist(Record_df_cp_Norm, method = "euclidean")) #same as p = 2

### k means
kmeans_Record_1<-NbClust::NbClust(Record_df_cp_Norm, 
                              min.nc=2, max.nc=4, method="kmeans")

table(kmeans_Record_1$Best.n[1,])

# Choose optimal k
# Silhouette Method
fviz_nbclust(Record_df_cp_Norm, method = "silhouette", 
             FUN = hcut, k.max = 4)+labs(subtitle = 'Silhouette Method')  ##gives k=2

# Elbow Method
fviz_nbclust(
  as.matrix(Record_df_cp_Norm), 
  kmeans, 
  k.max = 4,
  method = "wss",
  diss = get_dist(as.matrix(Record_df_cp_Norm), method = "manhattan")
)+labs(subtitle = 'Elbow Method') ##gives k=2

# Gap Statistics
fviz_nbclust(Record_df_cp_Norm,kmeans,k.max=4,method = 'gap_stat')+labs(subtitle = 'Gap Statistic Method') ##gives k=1


# k means 
kmeans_Record_1_Result<-kmeans(Record_df_cp_Norm,2,nstart=25)
# print the results
print(kmeans_Record_1_Result) 
kmeans_Record_1_Result$centers # view centers of 2 clusters

aggregate(Record_df_cp_Norm, 
          by=list(cluster=kmeans_Record_1_Result$cluster), mean) # view result with cluster list

summary(kmeans_Record_1_Result)

cbind(Record_df,cluster=kmeans_Record_1_Result$cluster) # place results in a table with original data

# see each cluster
kmeans_Record_1_Result$cluster

## see the size of each cluster
kmeans_Record_1_Result$size

## Visualize the clusters
fviz_cluster(kmeans_Record_1_Result, Record_df_cp_Norm, main="Euclidean")


### Hierarchical Clustering
(Dist_norm_M2<- dist(Record_df_cp_Norm, method = "minkowski", p=2)) #Euclidean
## run hclust
(HClust_Ward_Euc_N_Record <- hclust(Dist_norm_M2, method = "average" ))
plot(HClust_Ward_Euc_N_Record, cex=0.9, hang=-1, main = "Minkowski p=2 (Euclidean)")
rect.hclust(HClust_Ward_Euc_N_3D, k=2)

# Density based Clustering
db_record<-dbscan(as.matrix(Record_df_cp_Norm),0.5,2)
hullplot(as.matrix(Record_df_cp_Norm),db_record$cluster,main = 'Density Based Clustering-Record Data')


############################################# Text Data ############################################## 

# Load in documents from Corpus
TextCorpus<-Corpus(DirSource('Corpus'))
(getTransformations())
(ndocs<-length(TextCorpus))


## Convert to Document Term Matrix  and TERM document matrix

## DOCUMENT Term Matrix  (Docs are rows)
TextCorpus_DTM <- DocumentTermMatrix(TextCorpus,
                                      control = list(
                                        stopwords = TRUE, ## remove normal stopwords
                                        wordLengths=c(3, 10), ## get rid of words of len 2 or smaller or larger than 15
                                        removePunctuation = TRUE,
                                        removeNumbers = TRUE,
                                        tolower=TRUE
                                        #stemming = TRUE,
                                      ))

inspect(TextCorpus_DTM)

## TERM Document Matrix  (words are rows)
TextCorpus_TERM_DM <- TermDocumentMatrix(TextCorpus,
                                          control = list(
                                            stopwords = TRUE, ## remove normal stopwords
                                            wordLengths=c(3, 10), ## get rid of words of len 2 or smaller or larger than 15
                                            removePunctuation = TRUE,
                                            removeNumbers = TRUE,
                                            tolower=TRUE
                                            #stemming = TRUE,
                                          ))

inspect(TextCorpus_TERM_DM)


## Convert to DF 
TextCorpus_DF_DT <- as.data.frame(as.matrix(TextCorpus_DTM))
TextCorpus_DF_TermDoc <- as.data.frame(as.matrix(TextCorpus_TERM_DM))

## Convert to matrix 
TC_DTM_mat <- as.matrix(TextCorpus_DTM)
(TC_DTM_mat[1:10,1:8])

TC_TERM_Doc_mat <- as.matrix(TextCorpus_TERM_DM)
(TC_TERM_Doc_mat[1:8,1:10])

## WordCloud
word.freq <- sort(rowSums(TC_TERM_Doc_mat), decreasing = T)
wordcloud(words = names(word.freq), freq = word.freq*2, min.freq = 2,
          random.order = F)

# Distance Matrics
(Dist_t1<- dist(TextCorpus_DF_DT, method = "minkowski", p=1)) ##Manhattan
(Dist_t2<- dist(TextCorpus_DF_DT, method = "minkowski", p=2)) #Euclidean
(Dist_tE<- dist(TextCorpus_DF_DT, method = "euclidean")) #same as p = 2

### k means

# Choose optimal k
# Silhouette Method
fviz_nbclust(TextCorpus_DF_DT, method = "silhouette", 
             FUN = hcut, k.max = 9)+labs('Silhouette Method')  ##gives k=2

# Elbow Method
fviz_nbclust(
  as.matrix(TextCorpus_DF_DT), 
  kmeans, 
  k.max = 3,
  method = "wss",
  diss = get_dist(as.matrix(TextCorpus_DF_DT), method = "manhattan")
)+labs(subtitle = 'Elbow Method') ##gives k=2


# Gap Statistics
fviz_nbclust(TextCorpus_DF_DT,kmeans,k.max=3,method = 'gap_stat')+labs(subtitle = 'Gap Statistic Method') ##gives k=3

## kmeans on documents

kmeans_textcorp_Result <- kmeans(TC_DTM_mat, 3, nstart=25)   

# Print the results
print(kmeans_textcorp_Result)

kmeans_textcorp_Result$centers  

## Place results in a table with the original data
cbind(TextCorpus_DF_DT, cluster = t(kmeans_textcorp_Result$cluster))

## See each cluster
kmeans_textcorp_Result$cluster

# Cluster size
kmeans_textcorp_Result$size

## Visualize the clusters
fviz_cluster(kmeans_textcorp_Result,TextCorpus_DF_DT, 
             main="Euclidean", repel = TRUE)




# kmeans on words
kmeans_Textcorp_Result<-kmeans(t(TextCorpus_DF_DT),2,nstart = 4)

# print the results
print(kmeans_Textcorp_Result)
kmeans_Textcorp_Result$centers #view the centers


cbind(t(TextCorpus_DF_DT), cluster = kmeans_Textcorp_Result$cluster) # place results in a table with the original data

## See each cluster
kmeans_Textcorp_Result$cluster


# See cluster size
kmeans_Textcorp_Result$size


## Visualize the clusters
df_t<-t(TextCorpus_DF_DT)[,apply(t(TextCorpus_DF_DT),2,var,na.rm=T)!=0]
fviz_cluster(kmeans_Textcorp_Result, df_t, 
             main="Euclidean", repel = TRUE)


# Kmeans on documents
My_Kmeans_TextCorpD<-Kmeans(TextCorpus_DF_DT, centers=3 ,
                            method = "euclidean")
My_Kmeans_TextCorpD$cluster

fviz_cluster(My_Kmeans_TextCorpD, TextCorpus_DF_DT, 
             main="Euclidean k = 3",repel = TRUE) +
  scale_color_brewer('Cluster', palette='Set2') + 
  scale_fill_brewer('Cluster', palette='Set2') 


# Kmeans on words
My_Kmeans_TextCorp2<-Kmeans(df_t, centers=2 ,method = "euclidean")
fviz_cluster(My_Kmeans_TextCorp2, df_t, main="Euclidean k=2",repel = TRUE)

My_Kmeans_TextCorp3<-Kmeans(df_t, centers=2 ,method = "spearman")
fviz_cluster(My_Kmeans_TextCorp3, df_t, main="Spearman",repel = TRUE)


# Hierarchical Clustering
(Dist_CorpusM2<- dist(TextCorpus_DF_DT, method = "minkowski", p=2)) #Euclidean
## run hclust
(HClust_TextCorp <- hclust(Dist_CorpusM2, method = "ward.D" ))
plot(HClust_TextCorp, cex=0.9, hang=-1, main = "Minkowski p=2 (Euclidean)")
rect.hclust(HClust_TextCorp, k=3)


# Density based Clustering
db_text<-dbscan(as.matrix(TextCorpus_DF_DT),2.5,3)
hullplot(as.matrix(TextCorpus_DF_DT),db_text$cluster,main = 'Density Based Clustering-Text Data')
