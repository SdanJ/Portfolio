install.packages('arulesViz')
library(arulesViz)
library(igraph)
install.packages('networkD3')
library(networkD3)
library(visNetwork)


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

## Visualization
plot(ButterRules[1:10],method='graph',engine = 'htmlwidget') # create interactive visualizations with htmlwidget for Butter Rules(comes from support measure) 
plot(DiapersRules[1:10], method='grouped') # bubble plot for Diapers rules (comes from Confidence measure)
#plot(EggRules) 

## Network D3 for Egg Rules( comes from Lift measure)
trans_rules<-EggRules[1:10]

## Convert the RULES to a DATAFRAME
Rules_DF2<-DATAFRAME(trans_rules, separate = TRUE)
(head(Rules_DF2))
str(Rules_DF2)
## Convert to char
Rules_DF2$LHS<-as.character(Rules_DF2$LHS)
Rules_DF2$RHS<-as.character(Rules_DF2$RHS)

## Remove all {}
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[{]', replacement='')
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[}]', replacement='')

head(Rules_DF2) # check rules dataframe after removing all baskets

## Remove the sup, conf, and count
## USING LIFT
Rules_L<-Rules_DF2[c(1,2,5)]
names(Rules_L) <- c("SourceName", "TargetName", "Weight")
head(Rules_L,30)

## USING SUP
Rules_S<-Rules_DF2[c(1,2,3)]
names(Rules_S) <- c("SourceName", "TargetName", "Weight")
head(Rules_S,30)

## USING CONF
Rules_C<-Rules_DF2[c(1,2,4)]
names(Rules_C) <- c("SourceName", "TargetName", "Weight")
head(Rules_C,30)

## Choose and set -- Lift
Rules_Sup<-Rules_L


## Build the nodes 
(edgeList<-Rules_Sup)
MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE))

nodeList <- data.frame(ID = c(0:(igraph::vcount(MyGraph) - 1)), 
                       # because networkD3 library requires IDs to start at 0
                       nName = igraph::V(MyGraph)$name)
## Node Degree
(nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                       v = igraph::V(MyGraph), mode = "all")))

## Betweeness
BetweenNess <- igraph::betweenness(MyGraph, 
                                   v = igraph::V(MyGraph), 
                                   directed = TRUE) 

(nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))

## This can change the BetweenNess value if needed
BetweenNess<-BetweenNess/100

## Build the edges
getNodeID <- function(x){
  which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
}
(getNodeID("Beer,Coffee")) 

edgeList <- plyr::ddply(
  Rules_S, .variables = c("SourceName", "TargetName" , "Weight"), 
  function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                          TargetID = getNodeID(x$TargetName)))


#Calculate Dice similarities between all pairs of nodes
#The Dice similarity coefficient of two vertices is twice 
#the number of common neighbors divided by the sum of the degrees 
#of the vertices. Method dice calculates the pairwise Dice similarities 
#for some (or all) of the vertices. 
DiceSim <- igraph::similarity.dice(MyGraph, vids = igraph::V(MyGraph), mode = "all")
head(DiceSim)

#Create  data frame that contains the Dice similarity between any two vertices
F1 <- function(x) {data.frame(diceSim = DiceSim[x$SourceID +1, x$TargetID + 1])}
#Place a new column in edgeList with the Dice Sim
head(edgeList)
edgeList <- plyr::ddply(edgeList,
                        .variables=c("SourceName", "TargetName", "Weight", 
                                     "SourceID", "TargetID"), 
                        function(x) data.frame(F1(x)))
head(edgeList)

## Color
COLOR_P <- colorRampPalette(c("#00FF00", "#FF0000"), 
                            bias = nrow(edgeList), space = "rgb", 
                            interpolate = "linear")
COLOR_P
(colCodes <- COLOR_P(length(unique(edgeList$diceSim))))
edges_col <- sapply(edgeList$diceSim, 
                    function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
nrow(edges_col)

## NetworkD3 Object
D3_network_items <- networkD3::forceNetwork(
  Links = edgeList, # data frame that contains info about edges
  Nodes = nodeList, # data frame that contains info about nodes
  Source = "SourceID", # ID of source node 
  Target = "TargetID", # ID of target node
  Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship amongst nodes
  NodeID = "nName", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
  Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
  Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
  height = 700, # Size of the plot (vertical)
  width = 900,  # Size of the plot (horizontal)
  fontSize = 20, # Font size
  linkDistance = networkD3::JS("function(d) { return d.value*10; }"), # Function to determine distance between any two nodes, uses variables already defined in forceNetwork function (not variables from a data frame)
  linkWidth = networkD3::JS("function(d) { return d.value/10; }"),# Function to determine link/edge thickness, uses variables already defined in forceNetwork function (not variables from a data frame)
  opacity = 0.9, # opacity
  zoom = TRUE, # ability to zoom when click on the node
  opacityNoHover = 0.9, # opacity of labels when static
  linkColour = "red"   ###"edges_col"red"# edge colors
) 

# Plot network
#D3_network_Tweets

# Save network as html file
networkD3::saveNetwork(D3_network_items, 
                       "NetD3_items.html", selfcontained = TRUE)


