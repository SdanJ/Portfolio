#install.packages("SnowballC")
library(wordcloud)
library(SnowballC)
library(tm)
#install.packages('twitteR')
library(twitteR)
library(dplyr)
library(stringr)

consumer_key <- "nwSi4HuhlbpSzCqSx27GKOP6a"
consumer_secret <- "4YvzExkuHoliRqQkuxpDYG9F41wfOHrbYZXjvVMdeJ2XEeQRJY"
access_token <- "1243701223712227328-IibRtICnmvcJmFpxqeBgCrGFOqX7Vv"
access_secret <- "GXwEuhk4KDzciHyNKw3mRjCS4TsOsNMZ96pNTCtipdL1W"

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)
Sys.sleep(23)
tw = twitteR::searchTwitter('#一人暮らし', n = 10000,since = '2014-1-1')

tw_text<-sapply(tw, function(x) x$getText()) #identify and create text files 
tw_text_corpus<-Corpus(VectorSource(tw_text))#create a corpus from the collection of text files

#################################################### Before Cleaning ##################################################################################
# Show Japanese characters
par(family = "HiraKakuProN-W3")

tw_text_corpus

tw_before<-TermDocumentMatrix(tw_text_corpus)#build a term-document matrix
tw_before <- as.matrix(tw_before)
tw_before <- sort(rowSums(tw_before),decreasing=TRUE)
tw_before <- data.frame(word = names(tw_before),freq=tw_before)#convert words to dataframe
##The frequency table of words
head(tw_before,10)

##Plot word frequencies
barplot(tw_before[1:10,]$freq, las = 2, names.arg = tw_before[1:10,]$word, col ='blue', main ='Most frequent words', ylab = 'Word frequencies',encoding = "utf-8")

##Generate the Word Cloud
set.seed(1314562)
wordcloud(tw_text_corpus,min.freq=1,max.words=60,scale=c(3,1),,colors=brewer.pal(8, "Dark2"),random.color=T, random.order=F)

####################################################After Cleaning##################################################################################

##Data Cleaning on the text files

#remove punctuation
tw_text_corpus<-tm_map(tw_text_corpus,removePunctuation) 

# remove stopwords
tw_text_corpus<-tm_map(tw_text_corpus,function(x)removeWords(x,stopwords()))
tw_text_corpus<-tm_map(tw_text_corpus,removeWords,c('が','の','を','に','へ','と','から','より','で','ば','と','ので','から','けれども','のに','と','や','やら','たり','は','も','こそ','でも','しか','まで','ばかり','だけ','ほど','くらい','か','な','よ','ぞ','ぜ','わ','ね'))

# remove URL
removeURL <- function(x) gsub('http[[:alnum:]]*', '', x)
tw_text_corpus <- tm_map(tw_text_corpus, content_transformer(removeURL))

# remove Retweet
tw_text_corpus <- tm_map(tw_text_corpus, function(x)gsub("rt", "", x)) 
# remove at(@)
tw_text_corpus <- tm_map(tw_text_corpus, function(x)gsub("@\\w+", "", x))
# remove numbers/Digits
tw_text_corpus <- tm_map(tw_text_corpus, function(x)gsub("[[:digit:]]", "", x))  
# remove tabs
tw_text_corpus <- tm_map(tw_text_corpus, function(x)gsub("[ |\t]{2,}", "", x)) 
# remove blank spaces at the beginning
tw_text_corpus <- tm_map(tw_text_corpus, function(x)gsub("^ ", "", x)) 
# remove blank spaces at the end
tw_text_corpus <- tm_map(tw_text_corpus, function(x)gsub(" $", "", x))  




##The Corpus AFTER cleaning
tw_text_corpus


tw_after<-TermDocumentMatrix(tw_text_corpus)#build a term-document matrix
tw_after <- as.matrix(tw_after)
tw_after <- sort(rowSums(tw_after),decreasing=TRUE)
tw_after <- data.frame(word = names(tw_after),freq=tw_after)#convert words to dataframe
##The frequency table of words AFTER cleaning
head(tw_after,10)


##Plot word frequencies AFTER cleaning
barplot(tw_after[1:10,]$freq, las = 2, names.arg = tw_after[1:10,]$word, col ='blue', main ='Most frequent words', ylab = 'Word frequencies')

##Generate the Word Cloud AFTER cleaning
set.seed(1314562)
wordcloud(tw_text_corpus,min.freq=1,max.words=60,scale=c(3,1), colors=brewer.pal(8, "Dark2"),random.color=T, random.order=F)
