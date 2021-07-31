## Read in data
df<-read.csv('/Users/jinshengdan/Desktop/new.csv', stringsAsFactors=FALSE,
             fileEncoding="latin1")
## Check first 15 rows
head(df,15)

#### Check and Remove NA or NAN rows
nrow(df) # 318851
df<-na.omit(df)
nrow(df) # 159376
head(df,15)

#### Delete columns for url and ids
ncol(df) # 26
df<- subset (df, select = -c(url,id,Cid))
ncol(df) # 23
head(df)

##### Rename the column DOM(active days on market) as activeDays
colnames(df)[colnames(df) == "DOM"] <- "activeDays"
head(df)

##### Remove special symbols for column floor
df$floor<-gsub("[^[:digit:]]","",df$floor)
head(df)

##### Categorical columns
##### buildingType, renovationCondition, buildingStructure, elevator, fiveYearsProperty, subway
df$buildingType[df$buildingType == 1] <- 'tower'
df$buildingType[df$buildingType == 2] <- 'bungalow'
df$buildingType[df$buildingType == 3] <- 'combination of plate and tower'
df$buildingType[df$buildingType == 4] <- 'plate'
df
df<-df[-40,]
df

df$renovationCondition[df$renovationCondition == 1] <- 'other'
df$renovationCondition[df$renovationCondition == 2] <- 'rough'
df$renovationCondition[df$renovationCondition == 3] <- 'Simplicity'
df$renovationCondition[df$renovationCondition == 4] <- 'hardcover'

df$buildingStructure[df$buildingStructure == 1] <- 'unknow'
df$buildingStructure[df$buildingStructure == 2] <- 'mixed'
df$buildingStructure[df$buildingStructure == 3] <- 'brick and wood'
df$buildingStructure[df$buildingStructure == 4] <- 'brick and concrete'
df$buildingStructure[df$buildingStructure == 5] <- 'steel'
df$buildingStructure[df$buildingStructure == 6] <- 'steel-concrete composite'

df$elevator[df$elevator == 1] <- 'Elevator'
df$elevator[df$elevator == 0] <- 'No elevator'

df$fiveYearsProperty[df$fiveYearsProperty == 1] <- 'Five years property'
df$fiveYearsProperty[df$fiveYearsProperty == 0] <- 'No five years property'

df$subway[df$subway == 1] <- 'Subway'
df$subway[df$subway == 0] <- 'No subway'

df

write.csv(df,"/Users/jinshengdan/Desktop/cleaned_house.csv", row.names = FALSE)








