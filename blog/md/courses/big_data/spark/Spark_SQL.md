# Spark SQL Guide

## Intro
Aggregate functions in PySpark are functions that operate on a group of rows and return a single value. These functions are used in Spark SQL queries to summarize and analyze data.

Here are some examples of aggregate functions in PySpark:

Sample Data
from pyspark.sql import SparkSession

## create SparkSession
spark = SparkSession.builder.appName("AggregateFunctionsExample").getOrCreate()

## create a sample dataframe

```python
data = [("John", "Hardware", 1000),
("Sara", "Software", 2000),
("Mike", "Hardware", 3000),
("Lisa", "Hardware", 4000),
("David", "Software", 5000)]

df = spark.createDataFrame(data, ["name", "category", "sales"])
df.show()
```

This will create a dataframe with columns name, category, and sales and five rows of data. You can use this dataframe to apply the aggregate functions that were shown in the previous example.

1. SUM
   This function returns the sum of the values in a specified column.

```python

from pyspark.sql.functions import sum
df.select(sum("sales")).show()
```

# Output
+----------+
|sum(sales)|
+----------+
|    15000 |
+----------+

2. COUNT

This function returns the number of rows in a specified column.

```python
from pyspark.sql.functions import count
df.select(count("customer_id")).show()
```

# Output
+------------------+
|count(customer_id)|
+------------------+
|               5  |
+------------------+

3. AVG

This function returns the average value of a specified column.

```python
from pyspark.sql.functions import avg
df.select(avg("sales")).show()
```

# Output
+-------------------+
|         avg(sales)|
+-------------------+
|            3000.0 |
+-------------------+

5. MAX

This function returns the maximum value in a specified column.

```python
from pyspark.sql.functions import max
df.select(max("sales")).show()
```

# Output
+----------+
|max(sales)|
+----------+
|      5000|
+----------+

5. MIN

This function returns the minimum value in a specified column.

```python
from pyspark.sql.functions import min
df.select(min("sales")).show()
```

# Output
+----------+
|min(sales)|
+----------+
|      1000|
+----------+

6. GROUP BY

This function groups the data by one or more columns and then applies an aggregate function to each group.

```python
from pyspark.sql.functions import avg
df.groupBy("category").agg(avg("sales")).show()
```

# Output
+--------+----------+
|category|avg(sales)|
+--------+----------+
|Hardware|      2666|
|Software|      3500|
+--------+----------+

Note that the exact values might be different based on the data you are using. These are just a few examples of the many aggregate functions available in PySpark.

## Ref

- https://spark.apache.org/docs/latest/sql-programming-guide.html