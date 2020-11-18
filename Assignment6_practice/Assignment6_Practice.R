#install.packages("e1071")
library(e1071)  # for machine learning methods
#install.packages("mlr")
library(mlr)
#install.packages("caret")
library(caret)
#install.packages("naivebayes")
library(naivebayes)
library(datasets)
library(ggplot2)
library(MASS)  
library(stringr)
library(mclust)
library(cluster)
library(tm)

######################################
##          Record Data - iris      ##
######################################
data("iris")
head(iris)
str(iris)
(summary(iris))
(nrow(iris))


# Create Test and Train set
samplerownums<-sample(150,30) # create a sample of 30 numbers fro 1-150
(iris_Test<-iris[samplerownums,]) # Test set

# Remove and keep the labels for test set
(irisTestLabels<-iris_Test[,c(5)])
iris_Test<-iris_Test[,-c(5)]
head(iris_Test) # check test data after removing the label

# For training data, keep/have the class label
iris_Train<-iris[-samplerownums,]
head(iris_Train)

##########--------- SVM -----------##########
#------ Polynomial Kernel with cost of 0.1
SVM_fit_P<-svm(Species~., data=iris_Train,
               kernel='polynomial',cost=.1,
               scale=FALSE)
print(SVM_fit_P)

# Prediction
(pred_P<-predict(SVM_fit_P,iris_Test,type='class'))

# Confusion Matrix
(Ptable<-table(pred_P,irisTestLabels))

# We have 4 variables and so need our plot to be more precise
plot(SVM_fit_P, data=iris_Train, Petal.Width~Petal.Length, 
     slice=list(Sepal.Width=3, Sepal.Length=4))

# Misclassification Rate for Polynomial
(MR_P <- 1 - sum(diag(Ptable))/sum(Ptable))

#------- Linear Kernel with cost of 0.5
SVM_fit_L<-svm(Species~., data=iris_Train, 
               kernel="linear", cost=.5, 
               scale=FALSE)
print(SVM_fit_L)

# Prediction
(pred_L<-predict(SVM_fit_L,iris_Test,type='class'))

# Confusion Matrix
(Ltable<-table(pred_L,irisTestLabels))

# Plot
plot(SVM_fit_L, data=iris_Train, Petal.Width~Petal.Length, 
     slice=list(Sepal.Width=3, Sepal.Length=4))

## Misclassification Rate for Linear
(MR_L <- 1 - sum(diag(Ltable))/sum(Ltable))


#------- Radial Kernel with cost of 0.3
SVM_fit_R<-svm(Species~., data=iris_Train, 
               kernel="radial", cost=.3, 
               scale=FALSE)
print(SVM_fit_R)

# Prediction
(pred_R<-predict(SVM_fit_R,iris_Test,type='class'))

# Confusion Matrix
(Rtable<-table(pred_R,irisTestLabels))

# Plot
plot(SVM_fit_R, data=iris_Train, Petal.Width~Petal.Length,
     slice=list(Sepal.Width=3, Sepal.Length=4))

## Misclassification Rate for Radial
(MR_R <- 1 - sum(diag(Rtable))/sum(Rtable))


##########--------- Naive Bayes -----------##########
# Create Train/Test for Record Data
every7index<-seq(1,nrow(iris),7)
iris_test<-iris[every7index,]
iris_train<-iris[-every7index,]

# Remove labels from the Record data test set
(iris_test_Labels<-iris_test$Species)  # save the label to a save place
iris_test_nolabel<-subset(iris_test, select = -c(Species))
head(iris_test_nolabel)

# Apply Naive Bayes
NB_classfier<-naiveBayes(Species ~., data = iris_train, na.action=na.pass)
NB_Prediction<-predict(NB_classfier, iris_test_nolabel)
NB_classfier

# Check the prediction
(NB_Prediction)
# Confusion matrix
table(NB_Prediction,iris_test$Species)
# Plot the prediction
plot(NB_Prediction, main='Naive Bayes for iris dataset')



########################################
##     Text Data - Corpus 'CatBook'   ##
########################################

# Load in the documents (the corpus)
CatBookCorpus <- Corpus(DirSource("CatBook"))
(ndocs<-length(CatBookCorpus))

# The following will show you that you read in all the documents
(summary(CatBookCorpus))  ## This will list the docs in the corpus
(meta(CatBookCorpus[[1]])) ## meta data are data hidden within a doc - like id
(meta(CatBookCorpus[[1]],5))

# Change the Corpus into a DTM, a DF, and Matrix
# DocumnetTermMatrix
CatBook_dtm <- DocumentTermMatrix(CatBookCorpus,
                                 control = list(
                                   #stopwords = TRUE, ## remove normal stopwords
                                   wordLengths=c(4, 8), ## get rid of words of len 3 or smaller or larger than 15
                                   removePunctuation = TRUE,
                                   removeNumbers = TRUE,
                                   tolower=TRUE,
                                   #stemming = TRUE,
                                   remove_separators = TRUE
                                   #stopwords = MyStopwords,
                                   #removeWords(MyStopwords),
                                   #bounds = list(global = c(minTermFreq, maxTermFreq))
                                 ))
# Matrix
DTM_mat <- as.matrix(CatBook_dtm)
(DTM_mat[1:10,1:10])


# Look at word freuqncies out of interest
(WordFreq <- colSums(as.matrix(CatBook_dtm)))

(head(WordFreq))
(length(WordFreq))
ord <- order(WordFreq)
(WordFreq[head(ord)])
(WordFreq[tail(ord)])
# Row Sums
(Row_Sum_Per_doc <- rowSums((as.matrix(CatBook_dtm))))


# Copy of a matrix format of the data
CatBook_M <- as.matrix(CatBook_dtm)
(CatBook_M[1:10,1:5])

# Normalized Matrix of the data
CatBook_M_N1 <- apply(CatBook_M, 1, function(i) round(i/sum(i),5))
(CatBook_M_N1[1:10,1:5])
# NOTICE: Applying this function flips the data...see above.
# So, we need to TRANSPOSE IT (flip it back)  The "t" means transpose
CatBook_Matrix_Norm <- t(CatBook_M_N1)
(CatBook_Matrix_Norm[1:10,1:10])

# Have a look at the original and the norm to make sure
(CatBook_M[1:10,1:10])
(CatBook_Matrix_Norm[1:10,1:10])

# Convert to dataframe
CatBook_DF <- as.data.frame(as.matrix(CatBook_dtm))
(CatBook_DF[1:10, 1:4])
str(CatBook_DF)
(CatBook_DF$bright)
(nrow(CatBook_DF))  ## Each row is a document

# Convert a matrix (or normalized matrix) to a DF
CatBook_DF_From_Matrix_N<-as.data.frame(CatBook_Matrix_Norm)
(CatBook_DF_From_Matrix_N[1:10, 1:4])

# Find frequent words
(findFreqTerms(CatBook_dtm, 2500))
# Find associations with a selected conf
(findAssocs(CatBook_dtm, 'bright', 0.98))


# We have a dataframe of text data called CatBook_DF_From_Matrix_N
(CatBook_DF[1:10, 1:5])

# Get the row names
(DF_Row_Names <- row.names(CatBook_DF))

# New and empty list for the labels
MyNamesList <- c()
for(next_item in DF_Row_Names){
  Names <- strsplit(next_item, "_")
  Next_Name<- Names[[1]][1]
  if(Next_Name == "book"){
    Next_Name<-"Book"
  }
  if(Next_Name == 'cat'){
    Next_Name<-'Cat'
  }
  MyNamesList<-c(MyNamesList,Next_Name)
}

# Use the list of labels to bind together with your DF to created labeled data.
print(MyNamesList)

Labeled_DF_CatBook <- cbind(MyNamesList, CatBook_DF)
(Labeled_DF_CatBook[1:5, 1:5])

# Create a DF with no row names
rownames(Labeled_DF_CatBook) <- c()

# Check both
(Labeled_DF_CatBook[1:10, 1:5])

# Notice that the NAME OF THE LABEL is "MyNamesList"

Labeled_DF_CatBook$MyNamesList<-as.factor(Labeled_DF_CatBook$MyNamesList)

# Grabbing Every X value  
# This method works whether the data is in order or not.
X = 3   ## This will create a 1/3, 2/3 split. 
# Of course, X can be any number.
(every_X_index<-seq(1,nrow(Labeled_DF_CatBook),X))

# Use these X indices to make the Testing and then Training sets:
DF_Test<-Labeled_DF_CatBook[every_X_index, ]
DF_Train<-Labeled_DF_CatBook[-every_X_index, ]
## View the created Test and Train sets
(DF_Test[, 1:5])
(DF_Train[, 1:5])


## Make sure label is factor type

str(DF_Test$MyNamesList)  ## Notice that the label is called "MyNamesList" and is correctly set to type FACTOR. This is IMPORTANT!!
str(DF_Train$MyNamesList)  ## GOOD! also type FACTOR


## Copy the Labels
(DF_Test_Labels <- DF_Test$MyNamesList)
str(DF_Test_Labels)
## Remove the labels
DF_Test_NL<-DF_Test[ , -which(names(DF_Test) %in% c("MyNamesList"))]
(DF_Test[, 1:5])

## REMOVE THE LABEL FROM THE TRAINING SET...
DF_Train_Labels<-DF_Train$MyNamesList
DF_Train_NL<-DF_Train[ , -which(names(DF_Train) %in% c("MyNamesList"))]
(DF_Train_NL[, 1:5])

##########--------- SVM -----------##########
#------ Polynomial Kernel with cost of 0.1
SVM_text_P<-svm(MyNamesList~., data=DF_Train,
               kernel='polynomial',cost=.1,
               scale=FALSE)
print(SVM_text_P)


# Prediction
(pred_text_P<-predict(SVM_text_P, DF_Test_NL,type='class'))

# Confusion Matrix
(P_text_table<-table(pred_text_P,DF_Test_Labels))


# Plot
plot(SVM_text_P, DF_Train,world~bright)

# Misclassification Rate for Polynomial
(MR_text_P <- 1 - sum(diag(P_text_table))/sum(P_text_table))

#------- Linear Kernel with cost of 0.5
SVM_text_L<-svm(MyNamesList~., data=DF_Train, 
               kernel="linear", cost=.5, 
               scale=FALSE)
print(SVM_text_L)

# Prediction
(pred_text_L<-predict(SVM_text_L,DF_Test_NL,type='class'))

# Confusion Matrix
(L_text_table<-table(pred_text_L,DF_Test_Labels))

# Plot
plot(SVM_text_L, data=DF_Train, world~bright)

## Misclassification Rate for Linear
(MR_text_L <- 1 - sum(diag(L_text_table))/sum(L_text_table))


#------- Radial Kernel with cost of 0.3
SVM_text_R<-svm(MyNamesList~., data=DF_Train, 
               kernel="radial", cost=.3, 
               scale=FALSE)
print(SVM_text_R)

# Prediction
(pred_text_R<-predict(SVM_text_R,DF_Test_NL,type='class'))

# Confusion Matrix
(R_text_table<-table(pred_text_R,DF_Test_Labels))

# Plot
plot(SVM_text_R, data=DF_Train, world~bright)

## Misclassification Rate for Linear
(MR_text_R <- 1 - sum(diag(R_text_table))/sum(R_text_table))


########------- Naive Bayes ---------########

NB_text<-naiveBayes(DF_Train_NL, DF_Train_Labels, laplace = 1)
NB_text_Pred <- predict(NB_text, DF_Test_NL)
#NB_e1071_2
# confusion Matrix
table(NB_text_Pred,DF_Test_Labels)
(NB_text_Pred)

##Visualize
plot(NB_text_Pred,main='Naive Bayes for Text data')

