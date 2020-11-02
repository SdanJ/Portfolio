library(rpart)
#install.packages('rattle')
library(rattle)
#install.packages('rpart.plot')
library(rpart.plot)
library(RColorBrewer)
#install.packages('Cairo')
library(Cairo)


# Read in the data
DF=read.csv('/Users/jinshengdan/Desktop/JPN_.csv')
# Check the first certain rows of the data
head(DF)

# Remove the column of 'Year' since it has little to do with the DT here
DF<-subset(DF,select = -c(Year))
head(DF)


# Create Train/Test for Record data
(every6_indexes<-seq(1,nrow(DF),6))
(DF_Test<-DF[every6_indexes, ])
(DF_Train<-DF[-every6_indexes, ])


# Label here is called " Birth Rate>=8.9"
# Remove labels from the Record data test set
(DF_TestLabels<-DF_Test$Birth.Rate..8.9)
DF_Test<-subset( DF_Test, select = -c(Birth.Rate..8.9))
head(DF_Test)


# Apply Decision Tree
fitR <- rpart(DF_Train$Birth.Rate..8.9 ~+GDP..current.US..+GNI.per.capita.with.Atlas.method..current.US.., data = DF_Train, method="class")
summary(fitR)


# Predict the Test sets
predictedR= predict(fitR,DF_Test, type="class")
# Confusion Matrix
table(predictedR,DF_TestLabels)
# Visualization
fancyRpartPlot(fitR)


# Save the Decision Tree as a jpg image
jpeg("DecisionTree_JPN.jpg")
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

