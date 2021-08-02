library(shiny)
#install.packages("wordcloud2")
library(wordcloud2)
# install.packages("tm")
library(tm)
#install.packages("colourpicker")
library(colourpicker)
library(tidyverse)
#install.packages('shinycssloaders')
library(shinycssloaders)


ui <- fluidPage(
    
    style = 'color:#1E6D8C;font-size:16px;font-style:Arial',
    titlePanel("Word Cloud on Books"),
    # Create a container for tab panels
    
    # Create a "Word cloud" tab
    tabPanel(
        title = "Word cloud",
        style = 'color:#1E6D8C;font-size:16px;font-style:Arial',
        sidebarLayout(
            sidebarPanel(
                selectInput("selection",
                            label = "Choose a book:",
                            choices = c(
                                "The Prince" = "prince"
                            )
                ),
                
                hr(),
                sliderInput("num", "Maximum number of words",
                            value = 100, min = 5, max =300
                ),
                hr(),
                colourInput("col", "Color", value = "lightblue"),
                hr(),
                colourInput("bkcol", "Background color", value = "white"),
                hr(),
                radioButtons("shp", "Shape", choices = c('circle','cardioid','diamond','triangle-forward','triangle','pentagon','star'
                ))
                
            ),
            mainPanel(
                
                tabsetPanel(
                    
                    tabPanel("WordCloud",wordcloud2Output("cloud")%>% withSpinner(color="lightblue"), conditionalPanel(condition="$('html').hasClass('shiny-busy')",
                                                                                                                       tags$div("Loading...",style= 'color:lightblue;font-size:22px;font-style:Arial', id="loadmessage"))),
                    
                    tabPanel("Top 10 Words", plotOutput("bar")%>% withSpinner(color="pink"), conditionalPanel(condition="$('html').hasClass('shiny-busy')",
                                                                                                              tags$div("Loading...",style= 'color:pink;font-size:22px;font-style:Arial',id="loadmessage"))),
                    
                    tabPanel("Raw Text (partial)",textOutput("text")%>% withSpinner(color="#8244AB"), conditionalPanel(condition="$('html').hasClass('shiny-busy')",
                                                                                                             tags$div("Loading...",style= 'color:#8244AB;font-size:22px;font-style:Arial',id="loadmessage"))),
                    
                    tabPanel("Frequency Dataframe (partial)",dataTableOutput("df")%>% withSpinner(color="#1E6D8C"),conditionalPanel(condition="$('html').hasClass('shiny-busy')",
                                                                                                                          tags$div("Loading...",style= 'color:#1E6D8C;font-size:22px;font-style:Arial',id="loadmessage")))
                    
                    
                )
                
            )
            
            
        )
    )
)



server <- function(input, output) {
    data_source <- reactive({

        if (input$selection == "prince") {
            data <- readLines("The Prince.txt")
            
        }
        
        
        return(data)
    })
    
    
    
    turn_to_df<-function(data){
        if (is.character(data)) {
            corpus <- Corpus(VectorSource(data))
            corpus <- tm_map(corpus, tolower)
            corpus <- tm_map(corpus, removePunctuation)
            corpus <- tm_map(corpus, removeNumbers)
            corpus <- tm_map(corpus, removeWords, stopwords(tolower(input$language)))
            tdm <- as.matrix(TermDocumentMatrix(corpus))
            data <- sort(rowSums(tdm), decreasing = TRUE)
            data <- data.frame(word = names(data), freq = as.numeric(data))
            data <- subset(data, word != '“') 
            data <- subset(data, word != '”') 
            data<- data[order(-data$freq), ]
        }
        return(data)
    }
    
    create_wordcloud <- function(data, num_words = 100, color='lightblue', background = "white", shape = circle) {
        
        # If text is provided, convert it to a dataframe of word frequencies
        if (is.character(data)) {
            corpus <- Corpus(VectorSource(data))
            corpus <- tm_map(corpus, tolower)
            corpus <- tm_map(corpus, removePunctuation)
            corpus <- tm_map(corpus, removeNumbers)
            corpus <- tm_map(corpus, removeWords, stopwords(tolower(input$language)))
            tdm <- as.matrix(TermDocumentMatrix(corpus))
            data <- sort(rowSums(tdm), decreasing = TRUE)
            data <- data.frame(word = names(data), freq = as.numeric(data))
            data <- subset(data, word != '“') 
            data <- subset(data, word != '”') 
            
        }
        
        # Make sure a proper num_words is provided
        if (!is.numeric(num_words) || num_words < 3) {
            num_words <- 3
        }
        
        # Grab the top n most common words
        data <- head(data, n = num_words)
        if (nrow(data) == 0) {
            return(NULL)
        }
        
        wordcloud2(data, size=0.5, color= color,backgroundColor = background, shape = shape)
    }
    output$cloud <- renderWordcloud2({
        create_wordcloud(data_source(),
                         num_words = input$num,
                         color=input$col,
                         background = input$bkcol,
                         shape = input$shp
        )
    })
    
    output$text<-renderText(data_source()[1:150])
    
    output$df <-renderDataTable(turn_to_df(data_source())[1:100,])
    
    pal <- colorRampPalette(colors = c("pink", "orange"))(10)
    output$bar <-renderPlot({barplot(freq~word,turn_to_df(data_source())[1:10,],col = pal, 
                                     col.axis="pink")})
    
    
}

shinyApp(ui = ui, server = server)
