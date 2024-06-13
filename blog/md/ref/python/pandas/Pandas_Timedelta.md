# Timedelta in Pandas

Timedeltas are differences in times, expressed in different units such as days, hours, minutes, and seconds. They can be both positive and negative.

## Creating Timedelta Objects

### Using String Literal

```python
import pandas as pd

print(pd.Timedelta('2 days 2 hours 15 minutes 30 seconds'))
```

Output:

```
2 days 02:15:30
```

### Using Integer Value

```python
print(pd.Timedelta(6, unit='h'))
```

Output:

```
0 days 06:00:00
```

### Data Offsets

Data offsets like days, hours, minutes, seconds, etc., can be used to construct Timedelta objects.

```python
print(pd.Timedelta(days=2))
```

Output:

```
2 days 00:00:00
```

### Using `to_timedelta()`

```python
print(pd.to_timedelta(pd.Series(['1 days', '2 days 00:00:05'])))
```

Output:

```
0   1 days 00:00:00
1   2 days 00:00:05
dtype: timedelta64[ns]
```

## Operations

Operations can be performed on Series/DataFrames with timedelta objects.

### Addition Operations

```python
s = pd.Series(pd.date_range('2012-1-1', periods=3, freq='D'))
td = pd.Series([pd.Timedelta(days=i) for i in range(3)])
df = pd.DataFrame({'A': s, 'B': td})
df['C'] = df['A'] + df['B']

print(df)
```

Output:

```
           A      B          C
0 2012-01-01 0 days 2012-01-01
1 2012-01-02 1 days 2012-01-03
2 2012-01-03 2 days 2012-01-05
```

### Subtraction Operation

```python
df['D'] = df['C'] - df['B']

print(df)
```

Output:

```
           A      B          C          D
0 2012-01-01 0 days 2012-01-01 2012-01-01
1 2012-01-02 1 days 2012-01-03 2012-01-02
2 2012-01-03 2 days 2012-01-05 2012-01-03
```

--- 

These examples illustrate the creation and operations with Timedelta objects in Pandas. 