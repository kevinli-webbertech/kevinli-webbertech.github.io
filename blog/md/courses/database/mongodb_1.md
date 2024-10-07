# MongoDB - Literature and Installation

## What is MongoDB?

MongoDB is a popular NoSQL database known for its flexibility, scalability, and ease of use. Unlike traditional relational databases, MongoDB stores data in a flexible, JSON-like format, allowing for more dynamic schemas. In this guide, we'll explore the core concepts, methods, operations, and advanced features you need to know to work effectively with MongoDB.

MongoDB provides a document-oriented data model, meaning data is stored in BSON (Binary JSON) format. It allows for the storage of complex data types, making it highly suitable for modern applications that require flexible and scalable data storage solutions.

## Installation of Mongo

> Please install MongoDB Community Edition not Atlas version.

* Windows

[Windows installation](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/)

* Mac

[Mac installation](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/)

* Linux 

From a terminal, install gnupg and curl if they are not already available:

`sudo apt-get install gnupg curl`

To import the MongoDB public GPG key, run the following command:

`curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
--dearmor`

* Create a list file for MongoDB

* Create the list file /etc/apt/sources.list.d/mongodb-org-7.0.list for your version of Ubuntu.

```
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
```
* Reload local package database

`sudo apt-get update`

* Install the MongoDB packages

`sudo apt-get install -y mongodb-org`

![linux_installation1.png](../../../images/database/linux_installation1.png)

- https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/


## Installation of Dbeaver

Skipped. (Free version of Dbeaver does not support mongo connection.)

## Start Mongo Server (This is in ubuntu)

```shell
xiaofengli@xiaofenglx:~/Downloads$ sudo chown -R mongodb:mongodb /var/lib/mongodb
xiaofengli@xiaofenglx:~/Downloads$ sudo chown mongodb:mongodb /tmp/mongodb-27017.sock
xiaofengli@xiaofenglx:~/Downloads$ sudo service mongod restart
xiaofengli@xiaofenglx:~/Downloads$ sudo systemctl status mongod
```

In linux, it will look like this,

![start_mongodb.png](../../../images/database/start_mongodb.png)

Once we start the server like above, then we type,

`mongosh` 

to connect to your MongoDB. Once you successfully connect to MongoDB,
you will see the following terminal,

![connect_mongodb.png](../../../images/database/connect_mongodb.png)

## Takeaway

* `Mongod` is the server side service of linux. We will have to start it with root account.
* `Mongosh` is a client side CLI tools to connect to the MongoDB server.
  And this is where we write query code to manage data.

## Ref

- https://www.mongodb.com/docs/manual/tutorial/
- https://www.w3schools.com/mongodb/
- https://www.tutorialspoint.com/mongodb/index.htm