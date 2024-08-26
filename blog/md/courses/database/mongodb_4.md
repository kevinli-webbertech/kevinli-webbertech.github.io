# MongoDB programming with Python

Prerequisites

MongoDB stores data in JSON-like documents:
# Mongodb document (JSON-style)
document_1 = {
  "_id" : "BF00001CFOOD",
  "item_name" : "Bread",
  "quantity" : 2,
  "ingredients" : "all-purpose flour"
}

# python dictionary
dict_1 = {
  "item_name" : "blender",
  "max_discount" : "10%",
  "batch_number" : "RR450020FRG",
  "price" : 340
}

## Connecting Python and MongoDB Atlas

For the following tutorial, start by creating a 
virtual environment
, and activate it.

python -m venv env
source env/bin/activate

**Installation Python Packages**

Now that you are in your virtual environment, you can install PyMongo. In your terminal, type:

python -m pip install "pymongo[srv]"
python -m pip install python-dateutil

**Creating a MongoDB database in Python**

```python
from pymongo import MongoClient
def get_database():
 
   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   CONNECTION_STRING = "mongodb+srv://user:pass@cluster.mongodb.net/myFirstDatabase"
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(CONNECTION_STRING)
 
   # Create the database for our example (we will use the same database throughout the tutorial
   return client['user_shopping_list']
  
# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":   
  
   # Get the database
   dbname = get_database()
```


### Ref

https://www.mongodb.com/resources/languages/python