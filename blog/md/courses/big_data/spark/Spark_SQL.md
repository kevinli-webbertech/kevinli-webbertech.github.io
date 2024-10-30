# Spark SQL and Dateframe

## Intro

Spark SQL is a Spark module for structured data processing. Internally, Spark SQL uses this extra information to perform extra optimizations. When computing a result, the same execution engine is used, independent of which API/language you are using to express the computation.

Aggregate functions in PySpark are functions that operate on a group of rows and return a single value. These functions are used in Spark SQL queries to summarize and analyze data.

## create SparkSession and Dataframe

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("SQLExample").getOrCreate()
data = [("John", "Hardware", 1000),
("Sara", "Software", 2000),
("Mike", "Hardware", 3000),
("Lisa", "Hardware", 4000),
("David", "Software", 5000)]

df = spark.createDataFrame(data, ["name", "category", "sales"])
df.show()
```

This will create a dataframe with columns name, category, and sales and five rows of data. You can use this dataframe to apply the aggregate functions that were shown in the previous example.

## Spark SQL with aggregation functions

1. SUM
   This function returns the sum of the values in a specified column.

```python
from pyspark.sql.functions import sum
df.select(sum("sales")).show()
```

**Output**

```shell
+----------+
|sum(sales)|
+----------+
|    15000 |
+----------+
```

2. COUNT

This function returns the number of rows in a specified column.

```python
from pyspark.sql.functions import count
df.select(count("customer_id")).show()
```

**Output**

```shell
+------------------+
|count(customer_id)|
+------------------+
|               5  |
+------------------+
```

3. AVG

This function returns the average value of a specified column.

```python
from pyspark.sql.functions import avg
df.select(avg("sales")).show()
```

**Output**

```shell
+-------------------+
|         avg(sales)|
+-------------------+
|            3000.0 |
+-------------------+
```

5. MAX

This function returns the maximum value in a specified column.

```python
from pyspark.sql.functions import max
df.select(max("sales")).show()
```

**Output**

```shell
+----------+
|max(sales)|
+----------+
|      5000|
+----------+
```

5. MIN

This function returns the minimum value in a specified column.

```python
from pyspark.sql.functions import min
df.select(min("sales")).show()
```

**Output**

```shell
+----------+
|min(sales)|
+----------+
|      1000|
+----------+
```

6. GROUP BY

This function groups the data by one or more columns and then applies an aggregate function to each group.

```python
from pyspark.sql.functions import avg
df.groupBy("category").agg(avg("sales")).show()
```

**Output**

```shell
+--------+----------+
|category|avg(sales)|
+--------+----------+
|Hardware|      2666|
|Software|      3500|
+--------+----------+
```

Note that the exact values might be different based on the data you are using. These are just a few examples of the many aggregate functions available in PySpark.

## Usage of SQL

One use of Spark SQL is to execute SQL queries. Spark SQL can also be used to read data from an existing Hive installation. For more on how to configure this feature, please refer to the Hive Tables section. When running SQL from within another programming language the results will be returned as a Dataset/DataFrame. You can also interact with the SQL interface using the command-line or over JDBC/ODBC.

## SQL Lab1

* Step 1 Make a source code like this,

```python
 from pyspark.sql import SparkSession
 
 df = spark.read.json("examples/people.json")
 df.show()
 spark.stop()
```

* Step 2 Download spark tarball, unzip it and get the example jsons out

![download](download.png)

Exact the zip and you will see this,

![extracted_zip](extracted_zip.png)

This is my folder structures,

![folder1](folder1.png)

![folder2](folder2.png)

![code](code.png)

* Step 3 Run the code

![runcode](runcode.png)

* Step 4 update our code to the following,

Let us add more usage of the dataframe, once the data is imported into dataframe,
it is in our hand. In the following example, we use `df.select`, `df.filter` and `df.groupBy`

```python
from pyspark.sql import SparkSession

spark = SparkSession \
       .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
df = spark.read.json("examples/people.json")
df.show()
   
# spark, df are from the previous example
# Print the schema in a tree format
df.printSchema()
 
# Select only the "name" column
df.select("name").show()

# Select everybody, but increment the age by 1
df.select(df['name'], df['age'] + 1).show() 

# Select people older than 21
df.filter(df['age'] > 21).show()
  
 # Count people by age
df.groupBy("age").count().show()
spark.stop()
```

![new_code](new_code.png)

You can download the code from here [SQL1.py](kevinli-webbertech.github.io/blog/md/courses/big_data/spark/spark_code/SQL1.py)

## SQL Lab 2

Download the code from here [SQL2.py](kevinli-webbertech.github.io/blog/md/courses/big_data/spark/spark_code/SQL2.py). It is the same content as the following code,

```python
from pyspark.sql import SparkSession
spark = SparkSession \
       .builder \
       .appName("Python Spark SQL basic example") \
       .config("spark.some.config.option", "some-value") \
       .getOrCreate()
df = spark.read.json("examples/people.json")

# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("people")
sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()
spark.stop()
```

* Run the code

![run_sql2](run_sql2.png)

## Datasets and DataFrames

A Dataset is a distributed collection of data. Dataset is a new interface added in Spark 1.6 that provides the benefits of RDDs (strong typing, ability to use powerful lambda functions) with the benefits of Spark SQL’s optimized execution engine.

* A Dataset can be constructed from JVM objects and then manipulated using functional transformations (map, flatMap, filter, etc.).

* A DataFrame is a Dataset organized into named columns. It is conceptually equivalent to a table in a relational database or a data frame in R/Python, but with richer optimizations under the hood.

* DataFrames can be constructed from a wide array of sources such as: structured data files, tables in Hive, external databases, or existing RDDs.

* The Dataset API is available in Scala and Java. `Python does not have the support for the Dataset API`. But due to Python’s dynamic nature, many of the benefits of the Dataset API are already available (i.e. you can access the field of a row by name naturally row.columnName). The case for R is similar.

* The DataFrame API is available in Scala, Java, Python, and R. In Scala and Java, a DataFrame is represented by a Dataset of Rows. In the Scala API, DataFrame is simply a type alias of Dataset[Row]. While, in Java API, users need to use Dataset<Row> to represent a DataFrame.

## Ref

- https://spark.apache.org/docs/latest/sql-programming-guide.html
- https://sparkbyexamples.com/pyspark/pyspark-rdd-actions/
- https://spark.apache.org/sql/
- https://spark.apache.org/streaming/
- https://spark.apache.org/mllib/
- https://spark.apache.org/docs/latest/ml-guide.html