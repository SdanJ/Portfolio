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
