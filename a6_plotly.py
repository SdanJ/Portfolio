#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 15:53:48 2021

@author: jinshengdan
"""
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import pycountry




## Read in data
cd = pd.read_csv("/Users/jinshengdan/Desktop/country_wise_latest.csv")
pd.set_option('display.max_columns', None) ## show all the columns without ...
cd.head() 
cd.isnull().sum() 


#### Map

list_countries = cd['Country/Region'].unique().tolist()
# print(list_countries) # Uncomment to see list of countries
d_country_code = {}  # To hold the country names and their ISO
for country in list_countries:
    try:
        country_data = pycountry.countries.search_fuzzy(country)
        # country_data is a list of objects of class pycountry.db.Country
        # The first item  ie at index 0 of list is best fit
        # object of class Country have an alpha_3 attribute
        country_code = country_data[0].alpha_3
        d_country_code.update({country: country_code})
    except:
        print('could not add ISO 3 code for ->', country)
        # If could not find country, make ISO code ' '
        d_country_code.update({country: ' '})

# print(d_country_code) # Uncomment to check dictionary  

# create a new column iso_alpha in the df
# and fill it with appropriate iso 3 code
for k, v in d_country_code.items():
    cd.loc[(cd['Country/Region'] == k), 'iso_alpha'] = v
    
data = dict(type='choropleth', 
            locations = cd['iso_alpha'], 
            z = cd['Confirmed'], 
            text = cd['Country/Region'],colorscale = 'viridis')

layout = dict(title = 'Global Map - Number of Covid 19 Confirmed Cases',
              geo = dict( projection = {'type':'hammer'},
                         showlakes = True, 
                         lakecolor = 'rgb(0,118,255)'))
x = go.Figure(data = [data], 
              layout = layout)
plot(x)
x.write_html("/Users/jinshengdan/Desktop/map_plotly.html")

######################################################################################################
## Read in data
ss = pd.read_csv("/Users/jinshengdan/Desktop/responses.csv")
ss.head() 
ss.isnull().sum() 

### Bar Chart
fig = px.bar(ss, x='Now-Environment', color='Now-Environment', 
                title='Bar Chart - Distribution of Class Environment During Covid 19')

plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/bar.html")


### Density Hitmap
# FriendRelationships - The average change in the student's friend relationships (-3 - 3)
# FamilyRelationships - The average change in the student's family relationships (-3 - 3)
fig = px.density_heatmap(ss, x="FriendRelationships", y="FamilyRelationships", marginal_x="box", marginal_y="violin", title = 'Density Hitmap for Friend Relationships and FamilyRelationships')
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/densityhit.html")

######################################################################################################
## Read in data
yt = pd.read_csv("/Users/jinshengdan/Desktop/channels.csv")
pd.set_option('display.max_columns', None) ## show all the columns without ...
yt.head() ## look at first few rows

## Check number of rows
len(yt) # 104752

## Check missing values
yt.isnull().sum() 

##### CLEANING
## Remove the column location which contains nas only
## Remove columns that will not be used
yt = yt.drop('location', axis=1)
yt = yt.drop('description', axis=1)
yt = yt.drop('trailer_title', axis=1)
yt = yt.drop('title', axis=1)
yt = yt.drop('category_id', axis=1)
yt = yt.drop('channel_id', axis=1)
yt = yt.drop('picture_url', axis=1)
yt = yt.drop('profile_url', axis=1)
yt = yt.drop('trailer_url', axis=1)
## Remove rows where the enrtry of the column country and join_date is na
yt=yt.dropna(subset=["country","join_date"])
len(yt) # 70624

yt.isnull().sum() 
yt.shape # check number of rows and columns after cleaning


### Correlation Heatmap
yt_cor = yt.corr()
fig = px.imshow(yt_cor,title='Correlation Heatmap on YouTube Dataset')
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/corheat.html")



### Donut Chart for number of videos on each category
fig = px.pie(yt, values='videos', names='category_name',  color_discrete_sequence=px.colors.sequential.Agsunset, opacity=0.9, hole=0.5, title='Donut Chart - Number of Videos for Each Category')
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/donut.html")


### Scatter Plot for number of videos based on join date
fig = px.scatter(yt, x='join_date', y="videos",color_discrete_sequence=['orange'],title='Number of Videos by Join Date')
fig.update_xaxes(showspikes=True, spikecolor="olive", spikesnap="cursor", spikemode="across")
fig.update_yaxes(showspikes=True, spikecolor="lightsteelblue", spikethickness=2)
fig.update_layout(spikedistance=100, hoverdistance=100)
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/scatter.html")


########################################################################################################

###### Read in dataset

vw = pd.read_csv("/Users/jinshengdan/Desktop/Final Result.csv")
pd.set_option('display.max_columns', None) ## show all the columns without ...
vw.head() ## look at first few rows

vw['Views']=vw['Views'].str.replace("[^ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'-. ]", "") ## kepp the numbers only

## Remove columns that will not be used
vw = vw.drop('Video Link', axis=1)
vw.head() 

vw.shape


### Work on date, year
vw['Uploaded Date']=  vw['Uploaded Date'].str.replace("Streamed live on ", "")
vw['Uploaded Date'] = vw['Uploaded Date'].str.replace("Streamed live ", "")
vw['Uploaded Date'] = vw['Uploaded Date'].str.replace("Premiered on ", "")
vw['Uploaded Date'] = vw['Uploaded Date'].str.replace("Premiered ", "")

vw.head(17)
vw=vw.drop(vw.index[5])
vw=vw[vw['Uploaded Date'].str.contains('hours ago')== False]

# Defining a Function to convert a string to date
def CreateDate(InpString):
    import datetime
    return(datetime.datetime.strptime(InpString, '%d %b %Y'))
 
# Creating the Joining Date
vw['JoiningDate']=vw['Uploaded Date'].apply(CreateDate)
print(vw)
 
# Defining a function to get month
def getMonth(DOJ):
    return(DOJ.strftime('%B'))
 
# Defining a function to get year
def getYear(DOJ):
    return(DOJ.strftime('%Y'))
 
# Applying the month and year extractor functions
vw['JoiningMonth']=vw['JoiningDate'].apply(getMonth)
vw['JoiningYear']=vw['JoiningDate'].apply(getYear)

# 3d plot
fig = px.scatter_3d(vw, x="Likes on Video", y="Dislikes on Video", z="Views", color='JoiningMonth', 
                    symbol='JoiningMonth',title='Number of Views by Likes and Dislikes on Video across Months')
  
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/3d.html")


#######################################################################################################

#### Read in dataset
tp = pd.read_csv("/Users/jinshengdan/Desktop/top_500.csv")
pd.set_option('display.max_columns', None) ## show all the columns without ...
tp.head() ## look at first few rows

tp['Grade'].unique()

tp.isnull().sum() 
## Remove rows where the enrtry of the column Grade and Uploads is na
tp=tp.dropna(subset=["Grade","Uploads"])

tp.isnull().sum() 


### Violin plot for views by grade
fig = px.violin(tp, x="Grade", y="Views", color='Grade',
                 box=True,title='Views by Grade')  
# showing the plot
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/vio.html")


##### Mesh3d Plot 
fig = go.Figure(data=[go.Mesh3d(x=tp.Uploads, y=tp.Subscriptions, z=tp.Views,
                   alphahull=5,
                   opacity=0.3,
                   color='maroon')])
fig.update_layout(title='Mesh 3D Plot - Number of Views by Uploads and Subscriptions')

fig.update_layout(scene = dict(
    xaxis_title="Uploads",
    yaxis_title="Subscriptions",
    zaxis_title="Views")
)
fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=700
                  )
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/mesh3d.html")


##### Facet scatter plot
tp_sort= tp.sort_values(by ='Subscriptions' )
fig = px.scatter(tp_sort, x='Subscriptions', y="Views", color='Grade',
                 symbol='Grade',  facet_row='Grade',title='Distribution of Views by Subscriptions and Grade')

plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/facetscat.html")



##### Line chart
fig = px.line(tp, y="Uploads", line_dash='Grade',
              color='Grade',color_discrete_sequence= ['pink','purple','orange'], title='Distribution of Uploads across Grade')
  
plot(fig)
fig.write_html("/Users/jinshengdan/Desktop/line.html")


###################################################################################################################




