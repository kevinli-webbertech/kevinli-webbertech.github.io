# Descriptive Statistics with Pandas DataFrame

Pandas DataFrame provides a multitude of methods to compute descriptive statistics and other related operations. These methods can operate along different axes, similar to NumPy arrays.

We'll use a DataFrame to demonstrate various methods for descriptive statistics.

* DataFrame - "index"(axis=0, default),"columns"(axis=1)

```python
import pandas as pd
import numpy as np

# Create a Dictionary of series
d = {
    'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Smith', 'Jack', 'Lee', 'David', 'Gasper', 'Betina', 'Andres']),
    'Age': pd.Series([25, 26, 25, 23, 30, 29, 23, 34, 40, 30, 51, 46]),
    'Rating': pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8, 3.78, 2.98, 4.80, 4.10, 3.65])
}
```
### Creating a DataFrame

```
# Create a DataFrame
df = pd.DataFrame(d)
print(df)
```

Output:

```
     Name  Age  Rating
0     Tom   25    4.23
1   James   26    3.24
2   Ricky   25    3.98
3     Vin   23    2.56
4   Steve   30    3.20
5   Smith   29    4.60
6    Jack   23    3.80
7     Lee   34    3.78
8   David   40    2.98
9  Gasper   30    4.80
10 Betina   51    4.10
11 Andres   46    3.65
```

### Sum

Returns the sum of the values for the requested axis. By default, axis is index (axis=0).


```python
# Sum along index (axis=0)
print(df.sum())

# Sum along columns (axis=1)
print(df.sum(axis=1))
```

Output:  
**axis=0**
```
Age                     382                                382
Name     TomJamesRickyVinSteveSmithJackLeeDavidGasperBe...
Rating                  44.92                              44.92
dtype: object
```
**axis=1**  
```
0     29.23
1     29.24
2     28.98
3     25.56
4     33.20
5     33.60
6     26.80
7     37.78
8     42.98
9     34.80
10    55.10
11    49.65
dtype: float64
```

### Mean

Returns the average value.

```python
print(df.mean())
```

Output:

```
Age       31.833333
Rating     3.743333
dtype: float64
```

### Standard Deviation

Returns the Bressel standard deviation of the numerical columns.

```python
print(df.std())
```

Output:

```
Age       9.232682
Rating    0.661628
dtype: float64
```

### Descriptive Statistics Functions

Below are important functions for descriptive statistics:

| Function | Description                      |
|----------|----------------------------------|
| count()  | Number of non-null observations  |
| sum()    | Sum of values                    |
| mean()   | Mean of values                   |
| median() | Median of values                 |
| mode()   | Mode of values                   |
| std()    | Standard deviation of values     |
| min()    | Minimum value                    |
| max()    | Maximum value                    |
| abs()    | Absolute value                   |
| prod()   | Product of values                |
| cumsum() | Cumulative sum                   |
| cumprod()| Cumulative product               |

Note: DataFrame is a heterogeneous data structure. Generic operations don’t work with all functions.

* Functions like sum(), cumsum() work with both numeric and character (or) string data elements without any error. Though n practice, character aggregations are never used generally, these functions do not throw any exception.
* Functions like abs(), cumprod() throw exception when the DataFrame contains character or string data because such operations cannot be performed.


### Summarizing Data

The `describe()` function computes a summary of statistics pertaining to the DataFrame columns.

```python
print(df.describe())
```

Output:

```
              Age      Rating
count   12.000000   12.000000
mean    31.833333    3.743333
std      9.232682    0.661628
min     23.000000    2.560000
25%     25.000000    3.230000
50%     29.500000    3.790000
75%     35.500000    4.132500
max     51.000000    4.800000
```

The `describe()` function can exclude character columns and give a summary of numeric columns. To include specific types of columns:

```python
# Summarize string columns
print(df.describe(include=['object']))
```

Output:

```
       Name
count     12
unique    12
top    Ricky
freq       1
```

To include all columns together:

```python
print(df.describe(include='all'))
```

Output:

```
              Age  Name      Rating
count   12.000000    12   12.000000
unique        NaN    12         NaN
top           NaN  Ricky         NaN
freq          NaN     1         NaN
mean    31.833333    NaN    3.743333
std      9.232682    NaN    0.661628
min     23.000000    NaN    2.560000
25%     25.000000    NaN    3.230000
50%     29.500000    NaN    3.790000
75%     35.500000    NaN    4.132500
max     51.000000    NaN    4.800000
```

These examples and outputs demonstrate the use of various descriptive statistics functions in pandas.