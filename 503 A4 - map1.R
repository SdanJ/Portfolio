library(leaflet)
library(dplyr)

df<-read.csv('/Users/jinshengdan/Desktop/database.csv')

head(df)
summary(df)

### Map 1
china_lat <- 35
china_long <-103
#### circles with numbers - group surrounding locations
df %>%leaflet() %>% addTiles() %>%addProviderTiles(providers$CartoDB.DarkMatter)%>%
  setView(china_long,china_lat ,zoom = 5) %>% 
  addMarkers(lat=df$Latitude, lng=df$Longitude, 
             icon = list(iconUrl = 'https://th.bing.com/th/id/OIP.Uv9BlGyy6JENpSAnfoGvFgHaHa?pid=ImgDet&rs=1',
                          iconSize = c(50, 50)),
             clusterOptions = markerClusterOptions(),
             popup= paste(df$Type,
                          "<br><strong>Magnitude: </strong>", df$Magnitude,
                          "<br><strong>Depth: </strong>", df$Depth,
                          "<br><strong>Date: </strong>", df$Date,
                          "<br><strong>Time: </strong>", df$Time,
                          "<br><strong>Latitude: </strong>", df$Latitude,
                          "<br><strong>Longitude: </strong>", df$Longitude
                                             ))

