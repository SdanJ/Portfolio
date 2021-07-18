library(twitteR)
library(dplyr)
library(arules)
library(arulesViz)
library(igraph)

## Connect to Twitter
consumer_key <- 'ACvpBU23ZJKe5sxwpmN1830Jd'
consumer_secret <- 'Vxx6F9y1anjWJAp2kE3HeykIYKlTyt2n2le2ttCQJeEySsC26t'
access_token <- '1243701223712227328-vnIq43ZE5FlymWyjRwY4DD2OIhuTtK'
access_secret <- 'XqeDtfCCNS2TiulyXEEWtAVDwP6yftNUuVo9TM8CsrYr0'

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)
tw = twitteR::searchTwitter('#bestbooks', n = 300, since = '2021-1-1', lang = 'en')
d = twitteR::twListToDF(tw)
TransactionTweetsFile = 'TweetResults.csv'
(d$text[10])

## Start the file
Trans <- file(TransactionTweetsFile)
## Tokenize to words 
Tokens<-tokenizers::tokenize_words(
  d$text[1],stopwords = stopwords::stopwords("en"), 
  lowercase = TRUE,  strip_punct = TRUE, strip_numeric = TRUE,
  simplify = TRUE)

## Write tokens
cat(unlist(Tokens), "\n", file=Trans, sep=",")
close(Trans)

## Append remaining lists of tokens into file
## Recall - a list of tokens is the set of words from a Tweet

Trans <- file(TransactionTweetsFile, open = "a")
for(i in 2:nrow(d)){
  Tokens<-tokenizers::tokenize_words(d$text[i],stopwords = stopwords::stopwords("en"),
                                     lowercase = TRUE,  strip_punct = TRUE, simplify = TRUE)
  cat(unlist(Tokens), "\n", file=Trans, sep=",")
}
close(Trans)

## Read the transactions data into a dataframe
TweetDF <- read.csv(TransactionTweetsFile, 
                    header = FALSE, sep = ",")
head(TweetDF)
(str(TweetDF))

## Convert all columns to char 
TweetDF<-TweetDF %>%
  mutate_all(as.character)
(str(TweetDF))
# We can now remove certain words
TweetDF[TweetDF == "t.co"] <- ""
TweetDF[TweetDF == "rt"] <- ""
TweetDF[TweetDF == "http"] <- ""
TweetDF[TweetDF == "https"] <- ""

## Clean with grepl - every row in each column
MyDF<-NULL
MyDF2<-NULL
for (i in 1:ncol(TweetDF)){
  MyList=c() 
  MyList2=c() # each list is a column of logicals ...
  MyList=c(MyList,grepl("[[:digit:]]", TweetDF[[i]]))
  MyDF<-cbind(MyDF,MyList)  ## create a logical DF
  MyList2=c(MyList2,(nchar(TweetDF[[i]])<3))
  MyDF2<-cbind(MyDF2,MyList2) 
  ## TRUE is when a cell has a word that contains digits
}
## For all TRUE, replace with blank
TweetDF[MyDF] <- ""
TweetDF[MyDF2] <- ""
(head(TweetDF,10))

# Now we save the dataframe using the write table command 
write.table(TweetDF, file = "UpdatedTweetFile.csv", col.names = FALSE, 
            row.names = FALSE, sep = ",")
TweetTrans <- read.transactions("UpdatedTweetFile.csv", sep =",", 
                                format("basket"),  rm.duplicates = TRUE)
inspect(TweetTrans)

## Create the Rules
TweetTrans_rules = arules::apriori(TweetTrans,parameter = list(support=.001, conf=1, minlen=2))
##  Sort by Conf and check top 10 rules
SortedRules_conf <- sort(TweetTrans_rules, by="confidence", decreasing=TRUE)
inspect(SortedRules_conf[1:10])
## Sort by Sup and check top 10 rules
SortedRules_sup <- sort(TweetTrans_rules, by="support", decreasing=TRUE)
inspect(SortedRules_sup[1:10])
## Sort by Lift and check top 10 rules
SortedRules_lift <- sort(TweetTrans_rules, by="lift", decreasing=TRUE)
inspect(SortedRules_lift[1:10])

TweetTrans_rules<-SortedRules_lift[1:100]
inspect(TweetTrans_rules)
