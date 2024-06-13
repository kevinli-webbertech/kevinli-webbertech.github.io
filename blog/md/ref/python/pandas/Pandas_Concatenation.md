# Pandas Data Combination and Time Series Operations

## Combining Data with `pd.concat()`

Pandas provides the `pd.concat()` function to easily combine Series, DataFrame, or Panel objects along a specified axis.

```python
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False)
```

- **objs**: Sequence or mapping of objects to concatenate.
- **axis**: Axis to concatenate along.
- **join**: How to handle indexes on other axes.
- **ignore_index**: If True, do not use the index values on the concatenation axis.
- **join_axes**: Specific indexes to use for the other axes instead of performing inner/outer set logic.

### Examples:

- Concatenating two DataFrames:

```python
result = pd.concat([one, two])
```

- Concatenating with specified keys:

```python
result = pd.concat([one, two], keys=['x', 'y'])
```

- Concatenating with ignoring index:

```python
result = pd.concat([one, two], keys=['x', 'y'], ignore_index=True)
```

- Concatenating along axis=1:

```python
result = pd.concat([one, two], axis=1)
```

## Time Series Operations

Pandas offers a robust set of tools for working with time series data.

### Get Current Time:

```python
import pandas as pd
print(pd.datetime.now())
```

### Create a TimeStamp:

```python
print(pd.Timestamp('2017-03-01'))
```

### Create a Range of Time:

```python
print(pd.date_range("11:00", "13:30", freq="30min").time)
```

### Change the Frequency of Time:

```python
print(pd.date_range("11:00", "13:30", freq="H").time)
```

### Converting to Timestamps:

```python
print(pd.to_datetime(pd.Series(['Jul 31, 2009', '2010-01-10', None])))
```

---

These are the basic operations for combining data and working with time series in Pandas. 