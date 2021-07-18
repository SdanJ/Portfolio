#############################################################################################################
############################################    Network D3   ################################################
#############################################################################################################
## Use Network D3 to view results
## Convert the RULES to a DATAFRAME
Rules_DF2<-DATAFRAME(TweetTrans_rules, separate = TRUE)
(head(Rules_DF2))
str(Rules_DF2)
## Convert to char
Rules_DF2$LHS<-as.character(Rules_DF2$LHS)
Rules_DF2$RHS<-as.character(Rules_DF2$RHS)

## Remove all {}
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[{]', replacement='')
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[}]', replacement='')

head(Rules_DF2)

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

#### Standard Network D3
## Build a function to present Network D3 plot for three rules
std_d3<-function(rule){
  # build nodes
  (edgeList<-rule)
  MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE))
  
  nodeList <- data.frame(ID = c(0:(igraph::vcount(MyGraph) - 1)), 
                         # because networkD3 library requires IDs to start at 0
                         nName = igraph::V(MyGraph)$name)
  ## Node Degree
  (nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                         v = igraph::V(MyGraph), mode = "all")))
  
  ## Betweenness
  BetweenNess <- igraph::betweenness(MyGraph, 
                                     v = igraph::V(MyGraph), 
                                     directed = TRUE) 
  
  (nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))
  
  ## This can change the BetweenNess value if needed
  BetweenNess<-BetweenNess/100
  
  # build edges
  
  getNodeID <- function(x){
    which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
  }
  
  (getNodeID('staysafe'))
  
  edgeList <- plyr::ddply(
    rule, .variables = c("SourceName", "TargetName" , "Weight"), 
    function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                            TargetID = getNodeID(x$TargetName)))
  
  head(edgeList)
  nrow(edgeList)
  
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
  
  # color
  F2 <- colorRampPalette(c("#FFFF00", "#FF0000"), bias = nrow(edgeList), space = "rgb", interpolate = "linear")
  colCodes <- F2(length(unique(edgeList$diceSim)))
  edges_col <- sapply(edgeList$diceSim, function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
  #COLOR_P <- colorRampPalette(c("#4DB3E6", "#37004D"), 
  #                            bias = nrow(edgeList), space = "rgb", 
  #                            interpolate = "linear")
  #COLOR_P
  #(colCodes <- COLOR_P(length(unique(edgeList$diceSim))))
  #edges_col <- sapply(edgeList$diceSim, 
  #                    function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
  
  ## NetworkD3 Object
  D3_network_Tweets <- networkD3::forceNetwork(
    Links = edgeList, # data frame that contains info about edges
    Nodes = nodeList, # data frame that contains info about nodes
    Source = "SourceID", # ID of source node 
    Target = "TargetID", # ID of target node
    Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship amongst nodes
    NodeID = "nName", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
    Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
    Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
  ) 
  
  # Plot network
  #D3_network_Tweets
  
  # Save network as html file
  networkD3::saveNetwork(D3_network_Tweets, 
                         "std_NetworkD3_tw.html",selfcontained = TRUE)
  return(D3_network_Tweets)
}

std_d3(Rules_L)
std_d3(Rules_S)
std_d3(Rules_C)


### Fancy Network D3
## Build a function to present Network D3 plot for three rules
NETWORKD3<-function(rule){
  # build nodes
  (edgeList<-rule)
  MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE))
  
  nodeList <- data.frame(ID = c(0:(igraph::vcount(MyGraph) - 1)), 
                         # because networkD3 library requires IDs to start at 0
                         nName = igraph::V(MyGraph)$name)
  ## Node Degree
  (nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                         v = igraph::V(MyGraph), mode = "all")))
  
  ## Betweenness
  BetweenNess <- igraph::betweenness(MyGraph, 
                                     v = igraph::V(MyGraph), 
                                     directed = TRUE) 
  
  (nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))
  
  ## This can change the BetweenNess value if needed
  BetweenNess<-BetweenNess/100
  
  # build edges
  
  getNodeID <- function(x){
    which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
  }
  
  (getNodeID('staysafe'))
  
  edgeList <- plyr::ddply(
    rule, .variables = c("SourceName", "TargetName" , "Weight"), 
    function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                            TargetID = getNodeID(x$TargetName)))
  
  head(edgeList)
  nrow(edgeList)
  
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
  
  # color
  F2 <- colorRampPalette(c("#FFFF00", "#FF0000"), bias = nrow(edgeList), space = "rgb", interpolate = "linear")
  colCodes <- F2(length(unique(edgeList$diceSim)))
  edges_col <- sapply(edgeList$diceSim, function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
  #COLOR_P <- colorRampPalette(c("#4DB3E6", "#37004D"), 
  #                            bias = nrow(edgeList), space = "rgb", 
  #                            interpolate = "linear")
  #COLOR_P
  #(colCodes <- COLOR_P(length(unique(edgeList$diceSim))))
  #edges_col <- sapply(edgeList$diceSim, 
  #                    function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
  
  ## NetworkD3 Object
  D3_network_Tweets <- networkD3::forceNetwork(
    Links = edgeList, # data frame that contains info about edges
    Nodes = nodeList, # data frame that contains info about nodes
    Source = "SourceID", # ID of source node 
    Target = "TargetID", # ID of target node
    Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship amongst nodes
    NodeID = "nName", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
    Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
    Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
    height = 600, # Size of the plot (vertical)
    width = 1000,  # Size of the plot (horizontal)
    fontSize = 20, # Font size
    linkDistance = networkD3::JS("function(d) { return d.value*10; }"), # Function to determine distance between any two nodes, uses variables already defined in forceNetwork function (not variables from a data frame)
    linkWidth = networkD3::JS("function(d) { return d.value/10; }"),# Function to determine link/edge thickness, uses variables already defined in forceNetwork function (not variables from a data frame)
    opacity = 0.8, # opacity
    zoom = TRUE, # ability to zoom when click on the node
    opacityNoHover = 0.7, # opacity of labels when static
    linkColour = edges_col #edge colors
  ) 
  
  # Plot network
  #D3_network_Tweets
  
  # Save network as html file
  networkD3::saveNetwork(D3_network_Tweets, 
                         "NetworkD3_tw.html",selfcontained = TRUE)
  return(D3_network_Tweets)
}

NETWORKD3(Rules_L)
NETWORKD3(Rules_S)
NETWORKD3(Rules_C)
