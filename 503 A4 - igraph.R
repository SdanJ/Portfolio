#############################################################################################################
############################################    igraph   ################################################
#############################################################################################################
(edgeList<-Rules_L)
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
  Rules_L, .variables = c("SourceName", "TargetName" , "Weight"), 
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
head(nodeList)

### Reorder column -- move the column "nName" to the first position to apply igraph functions
new_nodelist = nodeList %>% select(nName, everything()) 

head(new_nodelist)

## Standard plot
ig_net<- graph_from_data_frame(d = edgeList, vertices = new_nodelist, directed = TRUE)
class(ig_net) # igraph

plot(ig_net,edge.arrow.size = 0.9,vertex.color="lightblue", main = 'Standard Network Plot')

# Community detection based on greedy optimization of modularity
cfg <- cluster_fast_greedy(as.undirected(ig_net))

plot(cfg, as.undirected(ig_net), main = 'Standard Community Detection Plot')


## Fancy plot
## Change the size of nodes and edges, the color of nodes and size of arrow, curved edge
V(ig_net)$size <- (V(ig_net)$nodeBetweenness+1)*3
V(ig_net)$label.color <- "tomato"
colrs <- c('','lightblue','','purple','','','','','','','','pink','','yellow')
V(ig_net)$color <- colrs[V(ig_net)$nodeDegree]
E(ig_net)$width <- as.numeric(E(ig_net)$Weight)*80
E(ig_net)$arrow.size <- .5
plot(ig_net,edge.color="orange",edge.curved=.2, vertex.label.dist=3, main = 'Network with Edited Layouts')

# Random layout
plot(ig_net, layout=layout_randomly(ig_net),edge.color="orange",edge.curved=.2, main = 'Random layout - Network')

# Circle layout
plot(ig_net, layout=layout_in_circle(ig_net),edge.color="orange",edge.curved=.2, main = 'Circle layout - Network')

# Force-directed layout
#plot(ig_net, layout=layout_with_fr(ig_net),edge.color="orange",edge.curved=.2, main = 'Force-directed layout - Network')

# Sphere layout
plot(ig_net, layout=layout_on_sphere(ig_net),edge.color="orange",edge.curved=.2, main = 'Sphere layout - Network')

# Community detection based on greedy optimization of modularity
plot(cfg, as.undirected(ig_net),edge.color="orange",edge.curved=.2, main = 'Community Detection with Edited Layout')



