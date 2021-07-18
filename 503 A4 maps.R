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
                                             ))%>% addGraticule(interval = 30, style = list(color = "grey", weight = 1))

### Map 2

df %>% leaflet() %>%
  setView(china_long,china_lat ,zoom = 6) %>%
  addProviderTiles("Esri.WorldStreetMap") %>%
  addCircles(
    lat=df$Latitude, lng=df$Longitude,
    radius = sqrt(10^df$Magnitude) * 10,
    color= ifelse(df$Magnitude>5.8,"orange","lightblue"),
    fillColor = ifelse(df$Magnitude>5.8,"red","lightgreen"),
    fillOpacity = 0.3,
    popup = paste(
      "<br><strong>Magnitude: </strong>", df$Magnitude,
      "<br><strong>Depth: </strong>", df$Depth,
      "<br><strong>Date: </strong>", df$Date,
      "<br><strong>Time: </strong>", df$Time,
      "<br><strong>Latitude: </strong>", df$Latitude,
      "<br><strong>Longitude: </strong>", df$Longitude
    )
  )%>%addMiniMap()%>%
  addLegend(labels=c("Magnitude > 5.8", "Magnitude < 5.8"), colors=c("red","lightblue"))


### Map 3

quakes <- df %>%
  dplyr::mutate(mag.level = cut(Magnitude,c(5,7,9,10),
                                labels = c('magnitude: >5 & <=7', 'magnitude: >7 & <=9','magnitude: >9')))

quakes.df <- split(quakes, quakes$mag.level)

l <- leaflet() %>% addTiles()

names(quakes.df) %>%
  purrr::walk(function(df) {
    l <<- l %>%
      addTiles(group = "OSM (default)") %>%
      addProviderTiles(providers$Stamen.Toner, group = "Toner") %>%
      addProviderTiles(providers$Esri.NatGeoWorldMap, group = "Esri") %>%
      setView(china_long,china_lat ,zoom = 2) %>%
      addMarkers(data=quakes.df[[df]],
                 lng=~Longitude, lat=~Latitude,
                 label=~as.character(Magnitude),
                 group = df,
                 popup = paste(
                   "<br><strong>Magnitude: </strong>", ~Magnitude,
                   "<br><strong>Depth: </strong>", ~Depth,
                   "<br><strong>Date: </strong>", ~Date,
                   "<br><strong>Time: </strong>", ~Time,
                   "<br><strong>Latitude: </strong>", ~Latitude,
                   "<br><strong>Longitude: </strong>", ~Longitude
                 ),
                 clusterOptions = markerClusterOptions(removeOutsideVisibleBounds = F),
                 labelOptions = labelOptions(noHide = F,
                                             direction = 'auto')
                )
  })

l %>%
  addLayersControl(
    baseGroups = c("OSM (default)", "Toner", "Esri"),
    overlayGroups = names(quakes.df),
    options = layersControlOptions(collapsed = FALSE)
  )
