install.packages('arulesViz')
library(arulesViz)

## First, check how data look as a dataframe
df<-read.csv('transaction.csv')
df

## Now, check the data in the form of transaction data
items<-read.transactions('transaction.csv',
                          rm.duplicates = FALSE, 
                          format = "basket",  ##if you use "single" also use cols=c(1,2)
                          sep=",",  ## csv file
                          cols=1) ## The dataset HAS row numbers
inspect(items)


## Apply apriori to the transaction data for Diapers rules
DiapersRules <- apriori(data=items,parameter = list(supp=.001, conf=.01, minlen=2),
                            appearance = list(default='lhs', rhs="Diapers"),
                            control=list(verbose=FALSE))
## Sort Diapers rules by the measure confidence and check top 10 rules
DiapersRules <- sort(DiapersRules, decreasing=TRUE, by="confidence")
inspect(DiapersRules[1:10])

## Apply apriori to the transaction data for Butter rules
ButterRules <- apriori(data=items,parameter = list(supp=.001, conf=.01, minlen=2),
                        appearance = list(default='lhs', rhs="Butter"),
                        control=list(verbose=FALSE))
## Sort Butter rules by the measure support and check top 10 rules
ButterRules <- sort(ButterRules, decreasing=TRUE, by="support")
inspect(ButterRules[1:10])


## Apply apriori to the transaction data for Egg rules
EggRules <- apriori(data=items,parameter = list(supp=.001, conf=.01, minlen=2),
                       appearance = list(default='lhs', rhs="Egg"),
                       control=list(verbose=FALSE))
## Sort Egg rules by the measure lift and check top 10 rules
EggRules <- sort(EggRules, decreasing=TRUE, by="lift")
inspect(EggRules[1:10])
