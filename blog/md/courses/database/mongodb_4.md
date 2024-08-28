# MongoDB programming with Python

This is a lab that we will write some python code with PyCharm and try to access the data from MongoDB via Python code.
Python code will help us do data analysis easier in advanced financial analysis.

This lab is optional and if you could reproduce it, it is a *bonus*.

## Prerequisites

* Install PyCharm (community version)
* Understand MongoDB document from our earlier class.

MongoDB stores data in JSON-like documents:

***Mongodb document (JSON-style)***

```json
document_1 = {
  "_id" : "BF00001CFOOD",
  "item_name" : "Bread",
  "quantity" : 2,
  "ingredients" : "all-purpose flour"
}
```

* Understand Python dictionary

The following `dict_1` is a dictionary.

```pythons
dict_1 = {
  "item_name" : "blender",
  "max_discount" : "10%",
  "batch_number" : "RR450020FRG",
  "price" : 340
}
```

* Understand how to start MongoDB server and how to connect to it successfully

![test_mongodb.png](test_mongodb.png)

* Understand that we could either use terminal to run and test python code or we could use PyCharm to do it.
In the following sections we will show you these two methods.

## Method 1: Connecting Python and MongoDB Using Terminals

For the following tutorial, start by creating a virtual environment, and activate it.
If you use IDE such as "Visual Studio Code", or Window|Linux|Mac Terminal, you could do the following,

**Step 1**. It will create a python virtual environment, so the package will be installed in a directory for you and it will avoid collisions with other Python versions.

`python -m venv env`

> `-m` is to specify a python module. Here we specify the module called `venv`. 
> `env` is a folder in your current directly. Once you execute it, please check if you can see this folder.

**Step 2**. Start and activate the virtual environment.

`source env/bin/activate`

After you execute the above two steps, you will see something like below,

![python_virtual_env.png](python_virtual_env.png)

**Installation Python Packages**

Now that you are in your virtual environment, you can install PyMongo. In your terminal, type:

`python -m pip install "pymongo[srv]"`

`python -m pip install python-dateutil`

After you execute the above two installation, you will see something similar to this,

![mongodb_py_packages.png](mongodb_py_packages.png)

### Lab Work

**Creating a MongoDB database in Python**

```python
from pymongo import MongoClient
import pprint

def get_database():

   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   CONNECTION_STRING = "mongodb://127.0.0.1:27017"

   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(CONNECTION_STRING)

   # Create the database for our example (we will use the same database throughout the tutorial
   return client['blog']

# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":

   # Get the database
   db= get_database()
   posts=db.posts
   pprint.pprint(posts.find_one())
```

To run it, we can see the following,

![mongodb_python_output.png](mongodb_python_output.png)

## Method2: Use Run Python code with PyCharm

If you use PyCharm, just create a new Python project and it will create the above virtual environment.

[TODO]

### Ref

- https://pymongo.readthedocs.io/en/stable/tutorial.html
- https://www.mongodb.com/resources/languages/python
