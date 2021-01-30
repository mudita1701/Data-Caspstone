
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:22:18 2021

@author: Puneet
"""
#importing all requiredl libraries and modules
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width',1000)
import matplotlib.cm as cm            # Matplotlib and associated plotting modules
import matplotlib.colors as colors
from sklearn.cluster import KMeans   # import k-means from clustering stage
import folium                        # map rendering library
from geopy.geocoders import Nominatim
import webbrowser
import json # library to handle JSON files
import requests # library to handle requests
from pandas.io.json import json_normalize


#h1 PART1- improting data for canada postal codes in the required format

#read data from the webpage
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
df1 = pd.read_html(url)
#check datatype for df1
type(df1)
df1
#It's a list and has unstructred data. Let's only capture data which is relevant
df2=df1[0:1]
type(df2)
df2
#to convert data into dataframe, first need to convert into nummpy array,then tranpose the list
df2_T0 = np.array(df2[0][0]).T.tolist()
df2_T1 = np.array(df2[0][1]).T.tolist()
df2_T2 = np.array(df2[0][2]).T.tolist()
#noq combine the list and convert into a dataframe
df3=pd.DataFrame(list(zip(df2_T0, df2_T1,df2_T2)),
              columns=['PostalCode','Borough','Neighbourhood'])
type(df3)
#drop frist row which has columns names
df4=df3.drop(df3.index[0])
#drop rows where borough='Not assigned'
df5=df4[df4.Borough != 'Not assigned']

#df5.to_csv("C:\\Users\\Puneet\\Desktop\\sample.csv")
#reset the index and check the value
df5.reset_index(drop=True,inplace=True)
df5
#check the shape of the dataframe
df5.shape


#h1 PART2- Getting Geocode informatin against each postal code

'''
import geocoder # import geocoder
# initialize your variable to None
lat_lng_coords = None

# loop until you get the coordinates
while(lat_lng_coords is None):
  g = geocoder.google('{}, Toronto, Ontario'.format('M3A'))
  lat_lng_coords = g.latlng

latitude = lat_lng_coords[0]
longitude = lat_lng_coords[1]'''

#getting Coordinates information of Toronto postal codes
Cordinates = pd.read_csv("C:\\Users\\Puneet\\Desktop\\Geospatial_Coordinates.csv")
Cordinates
#merging psotal codes infor with Coordinates informations
ToroGeo = df5.merge(Cordinates, how='inner', left_on=["PostalCode"], right_on=["Postal Code"])
ToroGeo
#drop column "Postal Code"
ToroGeo.drop(["Postal Code"], axis='columns',inplace=True)
ToroGeo

#h1 PART3- Explore and cluster the neighborhoods in Toronto

#getting cordinates for Toronto
address = 'Toronto'
geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))

# create map of Toronto using latitude and longitude values
map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=11)
# add markers to map
for lat, lng, label in zip(ToroGeo['Latitude'], ToroGeo['Longitude'],ToroGeo['Neighbourhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Toronto)  
map_Toronto.save("map_1.html") 
webbrowser.open("map_1.html")
    
# now clustering the neighborhoods in Toronto, Run _k_-means to cluster the neighborhood into 5 clusters.

#Define Foursquare Credentials and Version
CLIENT_ID = 'WFCKDZUDPQQAPN0HGJFHN2I3QILZYO4HELZ5CS41AEVMMP00' # your Foursquare ID
CLIENT_SECRET = 'BPKVDFNZLSLRN13414X4UKF1QBMWXM2PZLNXL3SXNKKO0WLK' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

ToroGeo.loc[0, 'Neighbourhood']
#Get the neighborhood's latitude and longitude values.
neighborhood_latitude = ToroGeo.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = ToroGeo.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = ToroGeo.loc[0, 'Neighbourhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))

#First, let's create the GET request URL. Name your URL **url**.
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL

#now request to get venue details
results = requests.get(url).json()
results


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
    
#Now we are ready to clean the json and structure it into a _pandas_ dataframe.
        venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
#And how many venues were returned by Foursquare?
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))



#Explore Neighborhoods in Toronto

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

#Now write the code to run the above function on each neighborhood and create a new dataframe called Toronto_venues.
# type your answer here
Torotno_venues = getNearbyVenues(names=ToroGeo['Neighbourhood'],
                                   latitudes=ToroGeo['Latitude'],
                                   longitudes=ToroGeo['Longitude']
                                  )

print(Torotno_venues.shape)
Torotno_venues.head(5)
#Let's check how many venues were returned for each neighborhood
Torotno_venues.groupby('Neighborhood').count()
#Let's find out how many unique categories can be curated from all the returned venues
print('There are {} uniques categories.'.format(len(Torotno_venues['Venue Category'].unique())))

#3. Analyze Each Neighborhood
# one hot encoding
Torotno_onehot = pd.get_dummies(Torotno_venues[['Venue Category']], prefix="", prefix_sep="")
Torotno_onehot
# add neighborhood column back to dataframe
Torotno_onehot['Neighborhood'] = Torotno_venues['Neighborhood'] 
# move neighborhood column to the first column
fixed_columns = [Torotno_onehot.columns[-1]] + list(Torotno_onehot.columns[:-1])
Torotno_onehot = Torotno_onehot[fixed_columns]

Torotno_onehot.head()
Torotno_onehot.shape

#Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
Torotno_grouped = Torotno_onehot.groupby('Neighborhood').mean().reset_index()
Torotno_grouped
Torotno_grouped.shape

#Let's print each neighborhood along with the top 5 most common venues
num_top_venues = 5
for hood in Torotno_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = Torotno_grouped[Torotno_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')

#Let's put that into a pandas dataframe
#First, let's write a function to sort the venues in descending order.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

#Now let's create the new dataframe and display the top 10 venues for each neighborhood.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Torotno_grouped['Neighborhood']

for ind in np.arange(Torotno_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Torotno_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
Torotno_grouped.head()

#4. Cluster Neighborhoods
# set number of clusters
kclusters = 5

Torotno_grouped_clustering = Torotno_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Torotno_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 

#manhattan_grouped_clustering
#Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Torotno_merged = ToroGeo

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
Torotno_merged = Torotno_merged.merge(neighborhoods_venues_sorted,how='inner',left_on=['Neighbourhood'], right_on=['Neighborhood'])

Torotno_merged.head() # check the last columns!

#Finally, let's visualize the resulting clusters

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Torotno_merged['Latitude'], Torotno_merged['Longitude'], Torotno_merged['Neighbourhood'], Torotno_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters.save("map_2.html") 
webbrowser.open("map_2.html")






