### Read in data
df<-read.csv('/Users/jinshengdan/Desktop/population_by_country_2020.csv')
head(df)

### Remove NA
nrow(df) #235
df<-na.omit(df)
nrow(df) #201

write.csv(df,'/Users/jinshengdan/Desktop/cleaned_pop.csv', row.names = FALSE)
