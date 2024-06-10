# Google MAP API

## Get a google map API key from google

## Check distance python example

`pip install googlemaps`, and run the following,

```
import googlemaps
from datetime import datetime

# Initialize the Google Maps client
gmaps = googlemaps.Client(key='YOUR_API_KEY')

# List of colleges in New Jersey
colleges = [
    "Atlantic Cape Community College",
    "Bergen Community College",
    "Bloomfield College",
    "Brookdale Community College",
    "Caldwell University",
    "Camden County College",
    "Centenary University",
    "College of Saint Elizabeth",
    "County College of Morris",
    "Cumberland County College",
    "Drew University",
    "Essex County College",
    "Fairleigh Dickinson University",
    "Felician University",
    "Georgian Court University",
    "Hudson County Community College",
    "Kean University",
    "Middlesex County College",
    "Monmouth University",
    "Montclair State University",
    "New Jersey City University",
    "New Jersey Institute of Technology",
    "Ocean County College",
    "Passaic County Community College",
    "Princeton University",
    "Ramapo College of New Jersey",
    "Raritan Valley Community College",
    "Rider University",
    "Rowan College at Burlington County",
    "Rowan College of South Jersey",
    "Rowan University",
    "Rutgers University",
    "Saint Peter's University",
    "Salem Community College",
    "Seton Hall University",
    "Stevens Institute of Technology",
    "Stockton University",
    "Sussex County Community College",
    "The College of New Jersey",
    "Thomas Edison State University",
    "Union County College",
    "Warren County Community College",
    "William Paterson University"
]

# Address to calculate distances from
origin = "22 Sherwood Lane, NJ, 07980"

# Calculate distances
distances = {}
for college in colleges:
    directions_result = gmaps.directions(origin, college, mode="driving")
    distance = directions_result[0]['legs'][0]['distance']['text']
    distances[college] = distance

# Print the distances
for college, distance in distances.items():
    print(f"{college}: {distance}")
```

Replace 'YOUR_API_KEY' with your actual Google Maps API key.