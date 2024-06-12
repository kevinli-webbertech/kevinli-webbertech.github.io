# Statistical Methods in Pandas

Statistical methods in Pandas help in understanding and analyzing the behavior of data. Let's explore a few statistical functions that can be applied to Pandas objects:

## Percent Change

The `pct_change()` function is available for Series, DataFrames, and Panels. It computes the percentage change between each element and its prior element.

#### Example:

```python
import pandas as pd

# Compute percentage change for a Series
s = pd.Series([10, 20, 30, 40, 50])
percent_change = s.pct_change()
print(percent_change)
```

```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'A': [10, 20, 30, 40, 50], 'B': [5, 10, 15, 20, 25]})

# Compute percentage change
percent_change_df = df.pct_change()
print(percent_change_df)
```

## Covariance

### Series:

The `cov()` method computes the covariance between series objects. NA values are automatically excluded.

#### Example:

```python
import pandas as pd

# Compute covariance for two Series
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([5, 4, 3, 2, 1])
covariance = s1.cov(s2)
print(covariance)
```

```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})

# Compute covariance
covariance_df = df.cov()
print(covariance_df)
```

## Correlation

Correlation measures the linear relationship between two arrays of values (series). Multiple methods like Pearson (default), Spearman, and Kendall can be used to compute correlation.

#### Example:

```python
import pandas as pd

# Compute correlation between two Series
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([5, 4, 3, 2, 1])
correlation = s1.corr(s2)
print(correlation)
```
```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})

# Compute correlation
correlation_df = df.corr()
print(correlation_df)
```
## Data Ranking

The `rank()` function produces ranking for each element in the array of elements. In case of ties, it assigns the mean rank.

### Parameters:

- `ascending`: (default True) When False, data is reverse-ranked, with larger values assigned a smaller rank.
- `method`: Tie-breaking method. Options include 'average', 'min', 'max', and 'first'.

#### Example:

```python
import pandas as pd

# Compute rank for a Series
s = pd.Series([10, 20, 30, 20, 10])
rank = s.rank(method='min', ascending=False)
print(rank)
```
```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'A': [10, 20, 30, 20, 10], 'B': [5, 10, 15, 10, 5]})

# Compute rank
rank_df = df.rank(method='min', ascending=False)
print(rank_df)
```

These statistical methods provide valuable insights into the data and are useful for various analytical tasks.