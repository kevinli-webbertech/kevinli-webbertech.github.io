## SQL Basics - MYSQL Database and SQL Literature

## Prerequisite

* MySQL 8 and Workbench installation
* DBeaver installation

## Outline

Get familiarized with basic SQL Programming.

* **Part I** Introduction to SQL, history and Literature

* **Part II** Introduction to MySQL Database
  * Create database
  * Drop database
  * Create user account
  * Grant user permissions

## Part I SQL History and Literature

### SQL History

SQL (Structured Query Language) is a domain-specific programming language used for managing and manipulating relational databases. It has a rich history that dates back to the 1970s and has evolved significantly over the years. Here’s an overview of the history of SQL:

1. Early Development (1970-1973)
   Relational Model: SQL’s history begins with the development of the relational model of data by Edgar F. Codd, an IBM researcher, in 1970. He published a seminal paper titled "A Relational Model of Data for Large Shared Data Banks," which introduced the concept of organizing data in tables (relations).
   SEQUEL: In 1973, a team at IBM's San Jose Research Laboratory, led by Donald D. Chamberlin and Raymond F. Boyce, developed a language called SEQUEL (Structured English Query Language) to manipulate and retrieve data stored in IBM’s experimental relational database system, System R. SEQUEL was designed to be more user-friendly and accessible to people without deep programming knowledge.

3. Renaming and Standardization (1974-1986)
   SEQUEL to SQL: Due to trademark issues, SEQUEL was renamed SQL. IBM continued to refine SQL, and it became a standard interface for relational databases.
   SQL/86: In 1986, the American National Standards Institute (ANSI) and the International Organization for Standardization (ISO) adopted SQL as the standard language for relational database management systems (RDBMS). This first standard was known as SQL-86 or SQL-1.
   Commercial Adoption: During this period, several commercial database systems that supported SQL emerged, including Oracle (1979), IBM’s DB2 (1983), and Microsoft SQL Server (1989).

4. Growth and Evolution (1987-1999)
   SQL/89: A minor revision of the standard, SQL-89, was released in 1989, introducing some clarifications and enhancements to SQL-86.
   SQL/92: The SQL-92 standard (also known as SQL2) was a significant update, introducing features like more complex queries, new data types, and enhanced language features. SQL-92 provided greater consistency across different database systems and helped solidify SQL’s dominance.
   SQL/99: SQL-99 (also known as SQL3) was another major revision that introduced object-relational database management system (ORDBMS) features, including support for structured types, recursive queries, triggers, and more advanced data handling capabilities.

5. Modern Enhancements (2000-Present)
   SQL:2003: This revision added XML-related features, standardized sequences, and introduced the "MERGE" statement, which allowed for conditional update/inserts.
   SQL:2006: This update focused on better integration with XML, reflecting the growing importance of XML in data exchange and storage.
   SQL:2008: Introduced new features like the TRUNCATE statement, improved diagnostics, and enhanced data types.
   SQL:2011: Introduced temporal databases, which allow the storage and querying of historical data (time periods).
   SQL:2016: Added features for JSON (JavaScript Object Notation) data management, reflecting the increasing use of JSON in modern applications.
   SQL:2019: Focused on enhanced support for big data and internet-of-things (IoT) applications, including support for polymorphic table functions and enhancements to the JSON capabilities.

6. SQL in Modern Context
   NoSQL Movement: In the 2000s, the NoSQL movement emerged as a response to the needs of web-scale applications, focusing on non-relational databases. Despite this, SQL remains the dominant language for relational databases.
   Open Source Databases: Open-source relational databases like MySQL (launched in 1995) and PostgreSQL (started in 1986, with its SQL implementation released in 1995) gained widespread popularity and contributed to SQL’s continued relevance.
   Cloud and Big Data: SQL has been adapted for cloud databases and big data platforms. Many NoSQL databases now support SQL-like query languages to bridge the gap between relational and non-relational data management.
   
**Legacy and Impact**
   SQL has had a profound impact on the way data is stored, retrieved, and managed across various industries. Its declarative nature, which allows users to specify what they want rather than how to get it, has made it accessible to a broad range of users, from developers to business analysts. Despite the rise of alternative database technologies, SQL continues to be a cornerstone of data management in the digital age.

## Installation

![dbeaver.png](../../../images/database/dbeaver.png)

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

## MySQL SELECT STATEMENT

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

## The MySQL INSERT INTO Statement

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

## MySQL NULL Values

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

## MySQL UPDATE Statement

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
