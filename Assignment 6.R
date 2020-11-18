library(e1071)  # for machine learning methods
library(mlr)
library(caret)
library(naivebayes)
library(datasets)
library(ggplot2)
library(MASS)  
library(stringr)
library(mclust)
library(cluster)
library(tm)

#########################################
##         Record Data - JPN.csv       ##
#########################################
Japan<-read.csv('JPN.csv')
head(Japan)
Japan<-Japan[,-c(1)]
str(Japan)
Japan$Birth.Rate..8.9<-as.factor(Japan$Birth.Rate..8.9)
Japan$GNI.per.capita.with.Atlas.method..current.US..<-as.numeric(Japan$GNI.per.capita.with.Atlas.method..current.US..)
str(Japan)
(summary(Japan))
(nrow(Japan))



# Create Test and Train set
samplerownums<-sample(50,16) # create a sample of 15 numbers fro 1-50
(JPN_Test<-Japan[samplerownums,]) # Test set

# Remove and keep the labels for test set
(JPNTestLabels<-JPN_Test[,c(4)])
JPN_Test<-JPN_Test[,-c(4)]
head(JPN_Test) # check test data after removing the label

# For training data, keep/have the class label
JPN_Train<-Japan[-samplerownums,]
head(JPN_Train)
str(JPN_Train)

##########--------- SVM -----------##########
#------- Polynomial Kernel with cost of 1
SVM_fit_P<-svm(Birth.Rate..8.9 ~., data=JPN_Train, 
               kernel="polynomial", cost=1, 
               degree=1, scale=FALSE)
print(SVM_fit_P)

# Prediction
(pred_P<-predict(SVM_fit_P,JPN_Test,type='class'))

# Confusion Matrix
(Ptable<-table(pred_P,JPNTestLabels))

# Plot
plot(SVM_fit_P,  data=JPN_Train, GNI.per.capita.with.Atlas.method..current.US..~GDP..current.US..,
     slice=list(Population.ages.65.and.above....of.total.population.=11.063263))

## Misclassification Rate for Polynomial
(MR_P <- 1 - sum(diag(Ptable))/sum(Ptable))


#------ Linear Kernel with cost of 0.1
SVM_fit_L<-svm(Birth.Rate..8.9 ~., data=JPN_Train,
               kernel='linear',cost=0.1,
               scale=FALSE)
print(SVM_fit_L)

# Prediction
(pred_L<-predict(SVM_fit_L,JPN_Test,type='class'))

# Confusion Matrix
(Ltable<-table(pred_L,JPNTestLabels))


# Plot
plot(SVM_fit_L, data=JPN_Train, GNI.per.capita.with.Atlas.method..current.US..~GDP..current.US..,
     slice=list(Population.ages.65.and.above....of.total.population.=11.063263))

## Misclassification Rate for Linear
(MR_L <- 1 - sum(diag(Ltable))/sum(Ltable))


#------- Radial Kernel with cost of 0.5
SVM_fit_R<-svm(Birth.Rate..8.9 ~., data=JPN_Train, 
               kernel="radial", cost=0.5, 
               scale=FALSE)
print(SVM_fit_R)

# Prediction
(pred_R<-predict(SVM_fit_R,JPN_Test,type='class'))

# Confusion Matrix
(Rtable<-table(pred_R,JPNTestLabels))

# Plot
plot(SVM_fit_R,  data=JPN_Train, GNI.per.capita.with.Atlas.method..current.US..~GDP..current.US..,
     slice=list(Population.ages.65.and.above....of.total.population.=11.063263))

## Misclassification Rate for Radial
(MR_R <- 1 - sum(diag(Rtable))/sum(Rtable))


##########--------- Naive Bayes -----------##########
# Create Train/Test for Record Data
every7index<-seq(1,nrow(Japan),7)
JPN_test<-Japan[every7index,]
JPN_train<-Japan[-every7index,]

# Remove labels from the Record data test set
(JPN_test_Labels<-JPN_test$Birth.Rate..8.9)  # save the label to a save place
JPN_test_nolabel<-subset(JPN_test, select = -c(Birth.Rate..8.9))
head(JPN_test_nolabel)

# Apply Naive Bayes
NB_classfier<-naiveBayes(Birth.Rate..8.9 ~., data = JPN_train, na.action=na.pass)
NB_Prediction<-predict(NB_classfier, JPN_test_nolabel)
NB_classfier

# Check the prediction
(NB_Prediction)
# Confusion matrix
table(NB_Prediction,JPN_test$Birth.Rate..8.9)
# Plot the prediction
plot(NB_Prediction, main='Naive Bayes for Japan dataset')


#################################################
##     Text Data - Corpus 'LiveAlone_Solo'     ##
#################################################
# Load in the documents (the corpus)
LiveSoloCorpus <- Corpus(DirSource("LiveAlone_Solo"))
(ndocs<-length(LiveSoloCorpus))

# The following will show you that you read in all the documents
(summary(LiveSoloCorpus))  ## This will list the docs in the corpus
(meta(LiveSoloCorpus[[1]])) ## meta data are data hidden within a doc - like id
(meta(LiveSoloCorpus[[1]],5))


# Change the Corpus into a DTM, a DF, and Matrix
# DocumnetTermMatrix
LiveSolo_dtm <- DocumentTermMatrix(LiveSoloCorpus,
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
DTM_mat <- as.matrix(LiveSolo_dtm)
(DTM_mat[1:10,1:10])

# Look at word freuqncies out of interest
(WordFreq <- colSums(as.matrix(LiveSolo_dtm)))

(head(WordFreq))
(length(WordFreq))
ord <- order(WordFreq)
(WordFreq[head(ord)])
(WordFreq[tail(ord)])

# Row Sums
(Row_Sum_Per_doc <- rowSums((as.matrix(LiveSolo_dtm))))


# Copy of a matrix format of the data
LiveSolo_M <- as.matrix(LiveSolo_dtm)
(LiveSolo_M[1:10,1:5])

# Normalized Matrix of the data
LiveSolo_M_N1 <- apply(LiveSolo_M, 1, function(i) round(i/sum(i),5))
(LiveSolo_M_N1[1:10,1:5])
# NOTICE: Applying this function flips the data...see above.
# So, we need to TRANSPOSE IT (flip it back)  The "t" means transpose
LiveSolo_Matrix_Norm <- t(LiveSolo_M_N1)
(LiveSolo_Matrix_Norm[1:10,1:10])

# Have a look at the original and the norm to make sure
(LiveSolo_M[1:10,1:10])
(LiveSolo_Matrix_Norm[1:10,1:10])

# Convert to dataframe
LiveSolo_DF <- as.data.frame(as.matrix(LiveSolo_dtm))
(LiveSolo_DF[1:10, 1:4])
str(LiveSolo_DF)
(LiveSolo_DF$夕飯)
(nrow(LiveSolo_DF))  ## Each row is a tweet

# Convert a matrix (or normalized matrix) to a DF
LiveSolo_DF_From_Matrix_N<-as.data.frame(LiveSolo_Matrix_Norm)
(LiveSolo_DF_From_Matrix_N[1:10, 1:4])


# Find frequent words
(findFreqTerms(LiveSolo_dtm, 2500))
# Find associations with a selected conf
(findAssocs(LiveSolo_dtm, '夕飯', 0.98))

# We have a dataframe of text data called LiveSolo_DF_From_Matrix_N
(LiveSolo_DF[1:10, 1:5])

# Get the row names
(DF_Row_Names <- row.names(LiveSolo_DF))

# New and empty list for the labels
MyNamesList <- c()
for(next_item in DF_Row_Names){
  Names <- strsplit(next_item, "_")
  Next_Name<- Names[[1]][1]
  MyNamesList<-c(MyNamesList,Next_Name)
}

# Use the list of labels to bind together with your DF to created labeled data.
print(MyNamesList)

Labeled_DF_LiveSolo <- cbind(MyNamesList, LiveSolo_DF)
(Labeled_DF_LiveSolo[1:5, 1:5])

# Create a DF with no row names
rownames(Labeled_DF_LiveSolo) <- c()

# Check both
(Labeled_DF_LiveSolo[1:10, 1:5])

# Notice that the NAME OF THE LABEL is "MyNamesList"

Labeled_DF_LiveSolo$MyNamesList<-as.factor(Labeled_DF_LiveSolo$MyNamesList)

# Grabbing Every X value  
# This method works whether the data is in order or not.
X = 3   ## This will create a 1/3, 2/3 split. 
# Of course, X can be any number.
(every_X_index<-seq(1,nrow(Labeled_DF_LiveSolo),X))

# Use these X indices to make the Testing and then Training sets:
DF_Test<-Labeled_DF_LiveSolo[every_X_index, ]
DF_Train<-Labeled_DF_LiveSolo[-every_X_index, ]
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
install.packages('RTextTools')
library(RTextTools)
# Configure the training data
nrow(LiveSolo_dtm)
container <- create_container(LiveSolo_dtm,DF_Test$MyNamesList, trainSize=1:100, virgin=FALSE)

# train a SVM Model
model <- train_model(container, "SVM", kernel="linear", cost=1)

#------ Polynomial Kernel with cost of 1
SVM_text_P<-svm(MyNamesList~., data=DF_Train,
                kernel='polynomial',cost=1,
                scale=FALSE)
print(SVM_text_P)

# Prediction
(pred_text_P<-predict(SVM_text_P, DF_Test_NL,type='class'))

# Confusion Matrix
(P_text_table<-table(pred_text_P,DF_Test_Labels))

# Plot
colnames(DF_Train)
plot(SVM_text_P, data=DF_Train,仕事~駅近,slice = list(不安=1,家族=1))

## Misclassification Rate for Polynomial
(MR_text_P <- 1 - sum(diag(P_text_table))/sum(P_text_table))

#------ Linear Kernel with cost of 0.1
SVM_text_L<-svm(MyNamesList~., data=DF_Train,
               kernel='linear',cost=0.1,
               scale=FALSE)
print(SVM_text_L)

# Prediction
(pred_text_L<-predict(SVM_text_L,DF_Test_NL,type='class'))

# Confusion Matrix
(L_text_table<-table(pred_text_L,DF_Test_Labels))


# Plot
plot(SVM_text_L, data=DF_Train,仕事~駅近,slice = list(不安=1,家族=1))

## Misclassification Rate for Linear
(MR_text_L <- 1 - sum(diag(L_text_table))/sum(L_text_table))


#------- Radial Kernel with cost of 0.5
SVM_text_R<-svm(MyNamesList~., data=DF_Train, 
               kernel="radial", cost=0.5, 
               scale=FALSE)
print(SVM_text_R)

# Prediction
(pred_text_R<-predict(SVM_text_R,DF_Test_NL,type='class'))

# Confusion Matrix
(R_text_table<-table(pred_text_R,DF_Test_Labels))

# Plot
plot(SVM_text_R, data=DF_Train,仕事~駅近,slice = list(不安=1,家族=1))

## Misclassification Rate for Radial
(MR_text_R <- 1 - sum(diag(R_text_table))/sum(R_text_table))


########------- Naive Bayes ---------########

NB_text<-naiveBayes(DF_Train_NL, DF_Train_Labels, laplace = 1)
NB_text_Pred <- predict(NB_text, DF_Test_NL)
#NB_e1071_2
# confusion Matrix
table(NB_text_Pred,DF_Test_Labels)
(NB_text_Pred)

##Visualize
plot(NB_text_Pred,main='Naive Bayes for Text data - LiveSolo')




