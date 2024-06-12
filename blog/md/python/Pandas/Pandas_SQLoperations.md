# Performing SQL Operations with Pandas

Pandas provides functionality to perform SQL-like operations on DataFrames, making it intuitive for users familiar with SQL. Let's explore some common SQL operations and their Pandas equivalents:

## 1. SELECT

In SQL, selection is done using a comma-separated list of columns or using `*` to select all columns:

```sql
SELECT total_bill, tip, smoker, time
FROM tips
LIMIT 5;
```

In Pandas, column selection is done by passing a list of column names to the DataFrame:

```python
import pandas as pd

url = 'https://raw.github.com/pandasdev/pandas/master/pandas/tests/data/tips.csv'
tips = pd.read_csv(url)
print(tips[['total_bill', 'tip', 'smoker', 'time']].head(5))
```

## 2. WHERE

Filtering in SQL is done via a WHERE clause:

```sql
SELECT * FROM tips WHERE time = 'Dinner' LIMIT 5;
```

In Pandas, filtering is done using Boolean indexing:

```python
import pandas as pd

url = 'https://raw.github.com/pandasdev/pandas/master/pandas/tests/data/tips.csv'
tips = pd.read_csv(url)
print(tips[tips['time'] == 'Dinner'].head(5))
```

## 3. GroupBy

SQL's GROUP BY operation can be used to fetch the count of records in each group:

```sql
SELECT sex, count(*)
FROM tips
GROUP BY sex;
```

The Pandas equivalent uses `groupby()` and `size()`:

```python
import pandas as pd

url = 'https://raw.github.com/pandasdev/pandas/master/pandas/tests/data/tips.csv'
tips = pd.read_csv(url)
print(tips.groupby('sex').size())
```

## 4. Top N Rows

SQL returns the top N rows using LIMIT:

```sql
SELECT * FROM tips
LIMIT 5;
```

Pandas provides a `head()` method to achieve the same:

```python
import pandas as pd

url = 'https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/tips.csv'
tips = pd.read_csv(url)
print(tips.head(5))
```

These Pandas operations provide a powerful and flexible way to manipulate data, similar to SQL queries.

