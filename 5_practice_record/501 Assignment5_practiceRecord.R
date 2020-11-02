library(rpart)
#install.packages('rattle')
library(rattle)
#install.packages('rpart.plot')
library(rpart.plot)
library(RColorBrewer)
#install.packages('Cairo')
library(Cairo)


# Read in the data
RecordDF=read.csv('titanic.csv')
# Check the first certain rows of the data
head(RecordDF)

# Take a small part of Titanic datasets
RecordDF = RecordDF[1:105,]

# Label here is called "survived"

# Create Train/Test for Record data
(every6_indexes<-seq(1,nrow(RecordDF),6))
(RecordDF_Test<-RecordDF[every6_indexes, ])
(RecordDF_Train<-RecordDF[-every6_indexes, ])


# Remove labels from the Record data test set
(RecordDF_TestLabels<-RecordDF_Test$survived)
RecordDF_Test<-subset( RecordDF_Test, select = -c(survived))
head(RecordDF_Test)


# Apply Decision Tree
fitR <- rpart(RecordDF_Train$survived ~sex+age+pclass, data = RecordDF_Train, method="class")
summary(fitR)


# Predict the Test sets
predictedR= predict(fitR,RecordDF_Test, type="class")
# Confusion Matrix
table(predictedR,RecordDF_TestLabels)
# Visualization
fancyRpartPlot(fitR)


# Save the Decision Tree as a jpg image
jpeg("DecisionTree_Titanic.jpg")
fancyRpartPlot(fitR)
dev.off()

# Information Gain with Entropy 
install.packages("CORElearn")
library(CORElearn)

Method.CORElearn <- CORElearn::attrEval(RecordDF_Train$survived ~ ., data=RecordDF_Train,  estimator = "InfGain")
(Method.CORElearn)
Method.CORElearn2 <- CORElearn::attrEval(RecordDF_Train$survived ~ ., data=RecordDF_Train,  estimator = "Gini")
(Method.CORElearn2)
Method.CORElearn3 <- CORElearn::attrEval(RecordDF_Train$survived ~ ., data=RecordDF_Train,  estimator = "GainRatio")
(Method.CORElearn3)

