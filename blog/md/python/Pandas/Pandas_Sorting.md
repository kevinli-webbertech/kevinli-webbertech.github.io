# Sorting in Pandas

In Pandas, there are two kinds of sorting available:

1. By label
2. By actual value

## By Label

### Using `sort_index()`

By using the `sort_index()` method, DataFrame can be sorted based on row labels in ascending order by default.

```python
import pandas as pd
import numpy as np

unsorted_df = pd.DataFrame(np.random.randn(10, 2), index=[1, 4, 6, 2, 3, 5, 9, 8, 0, 7], columns=['col2', 'col1'])
sorted_df = unsorted_df.sort_index()
print(sorted_df)
```

### Sorting Order

The order of sorting can be controlled by passing a Boolean value to the `ascending` parameter.

```python
sorted_df = unsorted_df.sort_index(ascending=False)
print(sorted_df)
```

## Sort the Columns

Sorting can also be done on column labels by passing the `axis` argument with a value of 0 or 1.

```python
sorted_df = unsorted_df.sort_index(axis=1)
print(sorted_df)
```

## By Value

### Using `sort_values()`

To sort by actual values, use the `sort_values()` method. It accepts a `by` argument, which specifies the column name of the DataFrame with which the values are to be sorted.

```python
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1')
print(sorted_df)
```

### Sorting with Multiple Columns

The `by` argument can take a list of column values to sort by multiple columns.

```python
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by=['col1', 'col2'])
print(sorted_df)
```

### Sorting Algorithm

The `sort_values()` method provides the option to choose the sorting algorithm from `mergesort`, `heapsort`, and `quicksort`. `mergesort` is the only stable algorithm.

```python
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1', kind='mergesort')
print(sorted_df)
```

These methods allow you to sort Pandas DataFrame by row or column labels, as well as by actual values, providing flexibility in data manipulation and analysis.