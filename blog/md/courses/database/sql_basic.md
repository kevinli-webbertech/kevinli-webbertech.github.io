## SQL Basics - MYSQL Databse

## Prerequisite

* MySQL 8 installation
* DBeaver installation

## Outline

Get familiarize with basic SQL Programming.

* Create database
* Drop database
* Create user account
* Grant user permissions
* Create table
* Alter table
* Insert, Select, Update, Delete Records
* Truncate table
* Delete table

## Log in to MySQL server

- Normal login to root of MySQL

`mysql -u root -p`

Then, enter your root password

## The MySQL CREATE DATABASE Statement
The CREATE DATABASE statement is used to create a new SQL database.

***Syntax***

`CREATE DATABASE databasename;`

***Example***

The following SQL statement creates a database called "testDB":

`CREATE DATABASE testDB;`

## CREATE Username and Password for the Database

After creating the database, the next step is to create a MySQL User:

`CREATE USER 'myUser'@'localhost' IDENTIFIED BY 'myPassword';`

***Note:***

* Replace myUser with your own username

* Replace myPassword with you own password

## Grant Privileges

`GRANT ALL PRIVILEGES ON databasename.* TO 'myUSER'@'localhost';`

***Note:***

* Replace databasename with your own database name that you created

* Replace myUser with you own username

### Flush Privileges

`FLUSH PRIVILEGES;`

Once a database is created, you can check it in the list of databases with the following SQL command:

`SHOW DATABASES;`

# MySQL DROP DATABASE Statement
The DROP DATABASE statement is used to drop an existing SQL database.

***Syntax***

`DROP DATABASE databasename;`

***Example***

The following SQL statement drops the existing database "testDB":

`DROP DATABASE testDB;`

## MySQL CREATE TABLE Statement

The CREATE TABLE statement is used to create a new table in a database.

***Syntax***

```
CREATE TABLE table_name (
    column1 datatype,
    column2 datatype,
    column3 datatype,
   ....
);
```
The column parameters specify the names of the columns of the table.

The datatype parameter specifies the type of data the column can hold (e.g. varchar, integer, date, etc.).

***Tip:*** For an overview of the available data types, go to our complete [Data Types Reference](https://www.w3schools.com/mysql/mysql_datatypes.asp)

***Example***

The following example creates a table called "Persons" that contains five columns: PersonID, LastName, FirstName, Address, and City:

```
CREATE TABLE Persons (
PersonID int,
LastName varchar(255),
FirstName varchar(255),
Address varchar(255),
City varchar(255)
);
```

The PersonID column is of type int and will hold an integer.

The LastName, FirstName, Address, and City columns are of type varchar and will hold characters, and the maximum length for these fields is 255 characters.

The empty "Persons" table will now look like this:

| PersonID | LastName | FirstName | Address | City   |
|----------|----------|-----------|---------|--------|
| &nbsp;   | &nbsp;   | &nbsp;    | &nbsp;  | &nbsp; |


### Create Table Using Another Table

A copy of an existing table can also be created using

`CREATE TABLE`

The new table gets the same column definitions. All columns or specific columns can be selected.

If you create a new table using an existing table, the new table will be filled with the existing values from the old table.

***Syntax***

```
CREATE TABLE new_table_name AS
    SELECT column1, column2,...
    FROM existing_table_name
    WHERE ....;
```

The following SQL creates a new table called "TestTables" (which is a copy of the "Customers" table):

```
CREATE TABLE TestTable AS
    SELECT customername, contactname
    FROM customers;
```

## The MySQL DROP TABLE Statement

The DROP TABLE statement is used to drop an existing table in a database.

***Syntax***

`DROP TABLE table_name;`

***Note:*** Be careful before dropping a table. Deleting a table will result in loss of complete information stored in the table!

***Example***
The following SQL statement drops the existing table "Shippers":

`DROP TABLE Shippers;`


## MySQL TRUNCATE TABLE
The TRUNCATE TABLE statement is used to delete the data inside a table, but not the table itself.

***Syntax***

`TRUNCATE TABLE table_name;`


## MySQL ALTER TABLE Statement

The ALTER TABLE statement is used to add, delete, or modify columns in an existing table.
The ALTER TABLE statement is also used to add and drop various constraints on an existing table.

### ALTER TABLE - ADD Column

To add a column in a table_name, use the following syntax:

```
ALTER TABLE table_name
ADD column_name datatype;
```

The following SQL adds an "Email" column to the "Customers" table:

```
ALTER TABLE Customers
ADD Email varchar(255);
```

### ALTER TABLE - DROP COLUMN

To delete a column in a table, use the following syntax (notice that some database systems don't allow deleting a column):

```
ALTER TABLE table_name
DROP COLUMN column_name;
```

The following SQL deletes the "Email" column from the "Customers" table:

```
ALTER TABLE Customers
DROP COLUMN Email;
```

### ALTER TABLE - MODIFY COLUMN

To change the datatype of a column in a table, use the following syntax:

```
ALTER TABLE table_name
MODIFY COLUMN column_name datatype;
```

### MySQL ALTER TABLE Example
Look at the "Persons" table:

| ID | LastName  | FirstName | Address      | City      |
|----|-----------|-----------|--------------|-----------|
| 1  | Hansen    | Ola       | Timoteivn 10 | Sandnes   |
| 2  | Svendson  | Tove      | Borgvn 23    | Sandnes   |
| 3  | Pettersen | Kari      | Storgt 20    | Stavanger |

Now we want to add a column named "DateOfBirth" in the "Persons" table. We use the following SQL statement:

***Example***

```
ALTER TABLE Persons
ADD DateOfBirth date;
```

Notice that the new column, "DateOfBirth", is of type date and is going to hold a date. The data type specifies what type of data the column can hold. For a complete reference of all the data types available in MySQL, go to our complete Data Types reference.

The "Persons" table will now look like this:

| ID  | LastName  | FirstName | Address      | City      | DateOfBirth |
|-----|-----------|-----------|--------------|-----------|-------------|
| 1   | Hansen    | Ola       | Timoteivn 10 | Sandnes   | &nbsp;      |
| 2   | Svendson  | Tove      | Borgvn 23    | Sandnes   | &nbsp;      |
| 3   | Pettersen | Kari      | Storgt 20    | Stavanger | &nbsp;      |


### Change Data Type Example

Now we want to change the data type of the column named "DateOfBirth" in the "Persons" table.

We use the following SQL statement:

***Example***

```
ALTER TABLE Persons
MODIFY COLUMN DateOfBirth year;
```

Notice that the "DateOfBirth" column is now of type year and is going to hold a year in a two- or four-digit format.

### DROP COLUMN Example
Next, we want to delete the column named "DateOfBirth" in the "Persons" table.

We use the following SQL statement:

***Example***

```
ALTER TABLE Persons
DROP COLUMN DateOfBirth;
```

The "Persons" table will now look like this:

| ID | LastName  | FirstName | Address      | City      |
|----|-----------|-----------|--------------|-----------|
| 1  | Hansen    | Ola       | Timoteivn 10 | Sandnes   |
| 2  | Svendson  | Tove      | Borgvn 23    | Sandnes   |
| 3  | Pettersen | Kari      | Storgt 20    | Stavanger |

# MySQL SELECT STATEMENT

The `SELECT` statement is used to select data from a database.

The data returned is stored in a result table, called the result-set.

### SELECT Syntax

```
SELECT column1, column2, ...
FROM table_name;
```

Here, column1, column2, ... are the field names of the table you want to select data from. If you want to select all the fields available in the table, use the following syntax:

`SELECT * FROM table_name;`


### Demo Database
In this tutorial we will use the well-known Northwind sample database.

Below is a selection from the "Customers" table in the Northwind sample database:

| CustomerID | CustomerName                       | ContactName        | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|--------------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                | Maria Anders       | Obere Str. 57                 | Berlin       | 12209      | Germany  |
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo       | Avda. de la Constitución 2222 | México D.F.  | 05021      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno     | Mataderos 2312                | México D.F.  | 05023      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy       | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |
| 5          | Berglunds snabbköp                 | Christina Berglund | Berguvsvägen 8                | Luleå        | S-958 22   | Sweden   |


### SELECT Columns Example

The following SQL statement selects the "CustomerName", "City", and "Country" columns from the "Customers" table:

***Example***
`SELECT CustomerName, City, Country FROM Customers;`


### SELECT * Example

The following SQL statement selects ALL the columns from the "Customers" table:

***Example***

```SELECT * FROM Customers;```

# The MySQL SELECT DISTINCT Statement

The `SELECT DISTINCT` statement is used to return only distinct (different) values.

Inside a table, a column often contains many duplicate values; and sometimes you only want to list the different (distinct) values.

### SELECT DISTINCT Syntax

```
SELECT DISTINCT column1, column2, ...
FROM table_name;
```

### SELECT Example Without DISTINCT
The following SQL statement selects all (including the duplicates) values from the "Country" column in the "Customers" table:

***Example***

`SELECT Country FROM Customers;`

Now, let us use the `SELECT DISTINCT` statement and see the result.


### SELECT DISTINCT Examples

The following SQL statement selects only the DISTINCT values from the "Country" column in the "Customers" table:

***Example***

`SELECT DISTINCT Country FROM Customers;`

The following SQL statement counts and returns the number of different (distinct) countries in the "Customers" table:

***Example***

`SELECT COUNT(DISTINCT Country) FROM Customers;`


## The MySQL WHERE Clause

The `WHERE` clause is used to filter records.

It is used to extract only those records that fulfill a specified condition.

### WHERE Syntax

```
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

***Note:*** The `WHERE` clause is not only used in `SELECT` statements, it is also used in `UPDATE`, `DELETE`, etc.!

### Demo Database
Below is a selection from the "Customers" table in the Northwind sample database:

| CustomerID | CustomerName                       | ContactName        | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|--------------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                | Maria Anders       | Obere Str. 57                 | Berlin       | 12209      | Germany  |
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo       | Avda. de la Constitución 2222 | México D.F.  | 05021      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno     | Mataderos 2312                | México D.F.  | 05023      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy       | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |
| 5          | Berglunds snabbköp                 | Christina Berglund | Berguvsvägen 8                | Luleå        | S-958 22   | Sweden   |



### WHERE Clause Example

The following SQL statement selects all the customers from "Mexico":

***Example***

```
SELECT * FROM Customers
WHERE Country = 'Mexico';
```

### Text Fields vs. Numeric Fields


SQL requires single quotes around text values (most database systems will also allow double quotes).

However, numeric fields should not be enclosed in quotes:

***Example***

```
SELECT * FROM Customers
WHERE CustomerID = 1;
```

### Operators in The WHERE Clause

The following operators can be used in the `WHERE` clause:

| Operator | Description                                                                  |
|----------|------------------------------------------------------------------------------|
| =        | Equal                                                                        |
| >        | Greater than                                                                 |
| <        | Less than                                                                    |
| >=       | Greater than or equal                                                        |
| <=       | Less than or equal                                                           |
| <>       | Not equal. Note: In some versions of SQL this operator may be written as !=  |
| BETWEEN  | Between a certain range                                                      |
| LIKE     | Search for a pattern                                                         |
| IN       | To specify multiple possible values for a column                             |


# The MySQL AND, OR and NOT Operators

The `WHERE` clause can be combined with `AND`, `OR`, and `NOT` operators.

The `AND` and `OR` operators are used to filter records based on more than one condition:

- The `AND` operator displays a record if all the conditions separated by `AND` are TRUE.
- The `OR` operator displays a record if any of the conditions separated by `OR` is TRUE.
- The `NOT` operator displays a record if the condition(s) is `NOT` TRUE.


### AND Syntax

```
SELECT column1, column2, ...
FROM table_name
WHERE condition1 AND condition2 AND condition3 ...;
```

### OR Syntax

```
SELECT column1, column2, ...
FROM table_name
WHERE condition1 OR condition2 OR condition3 ...;
```

### NOT Syntax

```
SELECT column1, column2, ...
FROM table_name
WHERE NOT condition;
```

### Demo Database
The table below shows the complete "Customers" table from the Northwind sample database:

| CustomerID | CustomerName                         | ContactName          | Address                              | City            | PostalCode | Country     |
|------------|--------------------------------------|----------------------|--------------------------------------|-----------------|------------|-------------|
| 1          | Alfreds Futterkiste                  | Maria Anders         | Obere Str. 57                        | Berlin          | 12209      | Germany     |
| 2          | Ana Trujillo Emparedados y helados   | Ana Trujillo         | Avda. de la Constitución 2222        | México D.F.     | 05021      | Mexico      |
| 3          | Antonio Moreno Taquería              | Antonio Moreno       | Mataderos 2312                       | México D.F.     | 05023      | Mexico      |
| 4          | Around the Horn                      | Thomas Hardy         | 120 Hanover Sq.                      | London          | WA1 1DP    | UK          |


### AND Example

The following SQL statement selects all fields from "Customers" where country is "Germany" AND city is "Berlin":

***Example***

```
SELECT * FROM Customers
WHERE Country = 'Germany' AND City = 'Berlin';
```

### OR Example

The following SQL statement selects all fields from "Customers" where city is "Berlin" OR "Stuttgart":

***Example***

```
SELECT * FROM Customers
WHERE City = 'Berlin' OR City = 'Stuttgart';
```

The following SQL statement selects all fields from "Customers" where country is "Germany" OR "Spain":

***Example***

```
SELECT * FROM Customers
WHERE Country = 'Germany' OR Country = 'Spain';
```

### NOT Example

The following SQL statement selects all fields from "Customers" where country is NOT "Germany":

***Example***

```
SELECT * FROM Customers
WHERE NOT Country = 'Germany';
```


### Combining AND, OR and NOT

You can also combine the `AND`, `OR` and `NOT` operators.

The following SQL statement selects all fields from "Customers" where country is "Germany" AND city must be "Berlin" OR "Stuttgart" (use parenthesis to form complex expressions):

***Example***

```
SELECT * FROM Customers
WHERE Country = 'Germany' AND (City = 'Berlin' OR City = 'Stuttgart');
```

The following SQL statement selects all fields from "Customers" where country is NOT "Germany" and NOT "USA":

***Example***

```
SELECT * FROM Customers
WHERE NOT Country = 'Germany' AND NOT Country = 'USA';
```


# The MySQL ORDER BY Keyword

The `ORDER BY` keyword is used to sort the result-set in ascending or descending order.

The `ORDER BY` keyword sorts the records in ascending order by default. To sort the records in descending order, use the `DESC` keyword.

### ORDER BY Syntax

```
SELECT column1, column2, ...
FROM table_name
ORDER BY column1, column2, ... ASC|DESC;
```

### Demo Database

Below is a selection from the "Customers" table in the Northwind sample database:

| CustomerID | CustomerName                        | ContactName         | Address                              | City           | PostalCode | Country    |
|------------|-------------------------------------|---------------------|--------------------------------------|----------------|------------|------------|
| 1          | Alfreds Futterkiste                 | Maria Anders        | Obere Str. 57                        | Berlin         | 12209      | Germany    |
| 2          | Ana Trujillo Emparedados y helados  | Ana Trujillo        | Avda. de la Constitución 2222        | México D.F.    | 05021      | Mexico     |
| 3          | Antonio Moreno Taquería             | Antonio Moreno      | Mataderos 2312                       | México D.F.    | 05023      | Mexico     |
| 4          | Around the Horn                     | Thomas Hardy        | 120 Hanover Sq.                      | London         | WA1 1DP    | UK         |
| 5          | Berglunds snabbköp                  | Christina Berglund  | Berguvsvägen 8                       | Luleå          | S-958 22   | Sweden     |


### ORDER BY Example

The following SQL statement selects all customers from the "Customers" table, sorted by the "Country" column:

***Example***

```
SELECT * FROM Customers
ORDER BY Country;
```

### ORDER BY DESC Example

The following SQL statement selects all customers from the "Customers" table, sorted DESCENDING by the "Country" column:

***Example***

```
SELECT * FROM Customers
ORDER BY Country DESC;
```

### ORDER BY Several Columns Example

The following SQL statement selects all customers from the "Customers" table, sorted by the "Country" and the "CustomerName" column. This means that it orders by Country, but if some rows have the same Country, it orders them by CustomerName:

***Example***

```
SELECT * FROM Customers
ORDER BY Country, CustomerName;
```

### ORDER BY Several Columns Example 2

The following SQL statement selects all customers from the "Customers" table, sorted ascending by the "Country" and descending by the "CustomerName" column:

***Example***

```
SELECT * FROM Customers
ORDER BY Country ASC, CustomerName DESC;
```


### The MySQL INSERT INTO Statement

The `INSERT INTO` statement is used to insert new records in a table.

### INSERT INTO Syntax

It is possible to write the `INSERT INTO` statement in two ways:

- Specify both the column names and the values to be inserted:

```
INSERT INTO table_name (column1, column2, column3, ...)
VALUES (value1, value2, value3, ...);
```

- If you are adding values for all the columns of the table, you do not need to specify the column names in the SQL query. However, make sure the order of the values is in the same order as the columns in the table. Here, the `INSERT INTO` syntax would be as follows:

```
INSERT INTO table_name
VALUES (value1, value2, value3, ...);
```

### Demo Database

Below is a selection from the "Customers" table in the Northwind sample database:

| CustomerID | CustomerName                        | ContactName         | Address                              | City           | PostalCode | Country    |
|------------|-------------------------------------|---------------------|--------------------------------------|----------------|------------|------------|
| 89         | White Clover Markets                | Karl Jablonski      | 305 - 14th Ave. S. Suite 3B          | Seattle        | 98128      | USA        |
| 90         | Wilman Kala                         | Matti Karttunen     | Keskuskatu 45                        | Helsinki       | 21240      | Finland    |
| 91         | Wolski                              | Zbyszek             | ul. Filtrowa 68                      | Walla          | 01-012     | Poland     |


### INSERT INTO Example

The following SQL statement inserts a new record in the "Customers" table:

***Example***

```
INSERT INTO Customers (CustomerName, ContactName, Address, City, PostalCode, Country)
VALUES ('Cardinal', 'Tom B. Erichsen', 'Skagen 21', 'Stavanger', '4006', 'Norway');
```

The selection from the "Customers" table will now look like this:

| CustomerID | CustomerName         | ContactName     | Address                     | City      | PostalCode | Country |
|------------|----------------------|-----------------|-----------------------------|-----------|------------|---------|
| 89         | White Clover Markets | Karl Jablonski  | 305 - 14th Ave. S. Suite 3B | Seattle   | 98128      | USA     |
| 90         | Wilman Kala          | Matti Karttunen | Keskuskatu 45               | Helsinki  | 21240      | Finland |
| 91         | Wolski               | Zbyszek         | ul. Filtrowa 68             | Walla     | 01-012     | Poland  |
| 92         | Cardinal             | Tom B. Erichsen | Skagen 21                   | Stavanger | 4006       | Norway  |


***Did you notice that we did not insert any number into the CustomerID field?***
The CustomerID column is an auto-increment field and will be generated automatically when a new record is inserted into the table.


### Insert Data Only in Specified Columns

It is also possible to only insert data in specific columns.

The following SQL statement will insert a new record, but only insert data in the "CustomerName", "City", and "Country" columns (CustomerID will be updated automatically):

***Example***

```
INSERT INTO Customers (CustomerName, City, Country)
VALUES ('Cardinal', 'Stavanger', 'Norway');
```

The selection from the "Customers" table will now look like this:

| CustomerID | CustomerName         | ContactName     | Address                     | City      | PostalCode | Country |
|------------|----------------------|-----------------|-----------------------------|-----------|------------|---------|
| 89         | White Clover Markets | Karl Jablonski  | 305 - 14th Ave. S. Suite 3B | Seattle   | 98128      | USA     |
| 90         | Wilman Kala          | Matti Karttunen | Keskuskatu 45               | Helsinki  | 21240      | Finland |
| 91         | Wolski               | Zbyszek         | ul. Filtrowa 68             | Walla     | 01-012     | Poland  |
| 92         | Cardinal             | null            | null                        | Stavanger | null       | Norway  |



# MySQL NULL Values

A field with a NULL value is a field with no value.

If a field in a table is optional, it is possible to insert a new record or update a record without adding a value to this field. Then, the field will be saved with a NULL value.

***Note:*** A NULL value is different from a zero value or a field that contains spaces. A field with a NULL value is one that has been left blank during record creation!

### How to Test for NULL Values?
It is not possible to test for NULL values with comparison operators, such as =, <, or <>.

We will have to use the IS `NULL` and IS `NOT NULL` operators instead.

### IS NULL Syntax

```
SELECT column_names
FROM table_name
WHERE column_name IS NULL;
```

### IS NOT NULL Syntax

```
SELECT column_names
FROM table_name
WHERE column_name IS NOT NULL;
```

### Demo Database

Below is a selection from the "Customers" table in the Northwind sample database:

| CustomerID | CustomerName                        | ContactName        | Address                       | City         | PostalCode | Country  |
|------------|-------------------------------------|--------------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                 | Maria Anders       | Obere Str. 57                 | Berlin       | 12209      | Germany  |
| 2          | Ana Trujillo Emparedados y helados  | Ana Trujillo       | Avda. de la Constitución 2222 | México D.F.  | 05021      | Mexico   |
| 3          | Antonio Moreno Taquería             | Antonio Moreno     | Mataderos 2312                | México D.F.  | 05023      | Mexico   |
| 4          | Around the Horn                     | Thomas Hardy       | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |
| 5          | Berglunds snabbköp                  | Christina Berglund | Berguvsvägen 8                | Luleå        | S-958 22   | Sweden   |



# MySQL UPDATE Statement

The `UPDATE` statement is used to modify the existing records in a table.

### UPDATE Syntax

```
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
```

***Note:*** Be careful when updating records in a table! Notice the WHERE clause in the `UPDATE` statement. The `WHERE` clause specifies which record(s) that should be updated. If you omit the `WHERE` clause, all records in the table will be updated!

### Demo Database

Below is a selection from the "Customers" table in the Northwind sample database:

| CustomerID | CustomerName                       | ContactName    | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|----------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                | Maria Anders   | Obere Str. 57                 | Berlin       | 12209      | Germany  |
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo   | Avda. de la Constitución 2222 | México D.F.  | 05021      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno | Mataderos 2312                | México D.F.  | 05023      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy   | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |


### UPDATE Table

The following SQL statement updates the first customer (CustomerID = 1) with a new contact person and a new city.

***Example***

```
UPDATE Customers
SET ContactName = 'Alfred Schmidt', City = 'Frankfurt'
WHERE CustomerID = 1;
```

The selection from the "Customers" table will now look like this:

| CustomerID | CustomerName                       | ContactName    | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|----------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                | Alfred Schmidt | Obere Str. 57                 | Frankfurt    | 12209      | Germany  |
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo   | Avda. de la Constitución 2222 | México D.F.  | 05021      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno | Mataderos 2312                | México D.F.  | 05023      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy   | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |


### UPDATE Multiple Records

It is the `WHERE` clause that determines how many records will be updated.

The following SQL statement will update the PostalCode to 00000 for all records where country is "Mexico":

***Example***

```
UPDATE Customers
SET PostalCode = 00000
WHERE Country = 'Mexico';
```

The selection from the "Customers" table will now look like this:

| CustomerID | CustomerName                       | ContactName    | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|----------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                | Alfred Schmidt | Obere Str. 57                 | Frankfurt    | 12209      | Germany  |
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo   | Avda. de la Constitución 2222 | México D.F.  | 00000      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno | Mataderos 2312                | México D.F.  | 00000      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy   | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |


## Update Warning!
Be careful when updating records. If you omit the `WHERE` clause, ALL records will be updated!

***Example***

```
UPDATE Customers
SET PostalCode = 00000;
```

The selection from the "Customers" table will now look like this:

| CustomerID | CustomerName                       | ContactName    | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|----------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                | Alfred Schmidt | Obere Str. 57                 | Frankfurt    | 00000      | Germany  |
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo   | Avda. de la Constitución 2222 | México D.F.  | 00000      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno | Mataderos 2312                | México D.F.  | 00000      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy   | 120 Hanover Sq.               | London       | 00000      | UK       |



# MySQL DELETE Statement
The `DELETE` statement is used to delete existing records in a table.

### DELETE Syntax

`DELETE FROM table_name WHERE condition;`

***Note:*** Be careful when deleting records in a table! Notice the `WHERE` clause in the `DELETE` statement. The `WHERE` clause specifies which record(s) should be deleted. If you omit the `WHERE` clause, all records in the table will be deleted!

### Demo Database

Below is a selection from the "Customers" table in the Northwind sample database:

| CustomerID | CustomerName                       | ContactName        | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|--------------------|-------------------------------|--------------|------------|----------|
| 1          | Alfreds Futterkiste                | Maria Anders       | Obere Str. 57                 | Berlin       | 12209      | Germany  |
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo       | Avda. de la Constitución 2222 | México D.F.  | 05021      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno     | Mataderos 2312                | México D.F.  | 05023      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy       | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |
| 5          | Berglunds snabbköp                 | Christina Berglund | Berguvsvägen 8                | Luleå        | S-958 22   | Sweden   |


### SQL DELETE Example

The following SQL statement deletes the customer "Alfreds Futterkiste" from the "Customers" table:

***Example***

`DELETE FROM Customers WHERE CustomerName='Alfreds Futterkiste';`

The "Customers" table will now look like this:

| CustomerID | CustomerName                       | ContactName        | Address                       | City         | PostalCode | Country  |
|------------|------------------------------------|--------------------|-------------------------------|--------------|------------|----------|
| 2          | Ana Trujillo Emparedados y helados | Ana Trujillo       | Avda. de la Constitución 2222 | México D.F.  | 05021      | Mexico   |
| 3          | Antonio Moreno Taquería            | Antonio Moreno     | Mataderos 2312                | México D.F.  | 05023      | Mexico   |
| 4          | Around the Horn                    | Thomas Hardy       | 120 Hanover Sq.               | London       | WA1 1DP    | UK       |
| 5          | Berglunds snabbköp                 | Christina Berglund | Berguvsvägen 8                | Luleå        | S-958 22   | Sweden   |

### Delete All Records

It is possible to delete all rows in a table without deleting the table. This means that the table structure, attributes, and indexes will be intact:

`DELETE FROM table_name;`

The following SQL statement deletes all rows in the "Customers" table, without deleting the table:

***Example***

`DELETE FROM Customers;`


___
