# Cassandra Introduction

## What is Cassandra?

Cassandra is a free and open-source, distributed, wide-column store, `NoSQL` database management system designed to handle large amounts of data across many commodity servers, providing high availability with no single point of failure. Cassandra offers support for clusters spanning multiple datacenters, with asynchronous masterless replication allowing low latency operations for all clients. Cassandra was designed to implement a combination of Amazon's Dynamo distributed storage and replication techniques combined with Google's Bigtable data and storage engine model.

### History

* `Avinash Lakshman`, one of the authors of Amazon's Dynamo, and `Prashant Malik` initially developed Cassandra at Facebook to power the Facebook inbox search feature.
* Facebook released Cassandra as an open-source project on Google code in July 2008. 
* In March 2009, it became an Apache Incubator project. 
* On February 17, 2010, it graduated to a top-level project.

### Create KeySpace in cassandra?

* KeySpace in NoSQL database is just like a schema in regular RDBMS concept, and it does not have any concrete structure. 
* In NoSQL database, there will be one keyspace per application. 
* A Keyspace contains column families or super columns. Each super column contains one or more column family, each column family contains at least one column.

## What are the Data Types in Cassandra?

In this tutorial, we will learn about the Data Types in `Cassandra CQL language`. 
DataTypes generally define the type of data a column can hold along with their own significance. 
CQL language supports the below list Data types:

* Native Types 
* Collection Types
* User-defined Types
* Tuple types

### Understanding Cqlsh in Cassandra

* CQLSH – This is the Command Line Utility used to execute the commands to communicate with Cassandra database.
* To start the utility we need to give the command cqlsh either in linux terminal or windows command prompt. 
* The default listen port for cqlsh is 9042

## Ref

- https://www.i2tutorials.com/cassandra-tutorial/cassandra-create-table/
- https://www.w3schools.blog/cassandra-tutorial