# Slicing and Dicing Pandas Data

We'll explore how to slice and dice data in Pandas, effectively obtaining subsets of Pandas objects like Series and DataFrames. While the Python and NumPy indexing operators `[ ]` and attribute operator `.` provide quick access to Pandas data structures, for more optimized access, Pandas offers specialized methods.

## Multi-Axes Indexing

Pandas supports three types of multi-axes indexing:

### 1. `.loc()`

Label-based indexing. It includes the start bound in slicing.

### 2. `.iloc()`

Integer-based indexing, akin to Python and NumPy's 0-based indexing.

### 3. `.ix()`

A hybrid method supporting both label and integer-based indexing. (Deprecated in newer versions of Pandas)

### .loc()

#### Example:

```python
import pandas as pd

# Accessing a single label
print(df.loc['row_label', 'col_label'])

# Accessing a list of labels
print(df.loc[['row1', 'row2'], ['col1', 'col2']])

# Accessing a slice object
print(df.loc['row1':'row3', 'col1':'col3'])

# Accessing with boolean array
print(df.loc[df['col1'] > 0])
```

### .iloc()

#### Example:

```python
import pandas as pd

# Accessing by integer
print(df.iloc[0, 1])

# Accessing with a list of integers
print(df.iloc[[0, 1], [0, 1]])

# Accessing with a range of values
print(df.iloc[0:3, 0:3])
```

### .ix()

#### Example (Deprecated in newer versions):

```python
import pandas as pd

# Accessing with a mix of labels and integers
print(df.ix[0:3, 'col1'])
```

## Notations and Return Types

When using multi-axes indexing, the following notations are used:

- For Series, `.loc[indexer]` returns a scalar value.
- For DataFrames, `.loc[row_index, col_index]` returns a Series object.
- For Panels (deprecated), `.loc[item_index, major_index, minor_index]` returns a Panel object.

Additionally, `.iloc()` and `.ix()` apply the same indexing options and return types.

## Attribute Access

Columns can also be selected using the attribute operator `.`.

#### Example:

```python
import pandas as pd

# Accessing a column using attribute access
print(df.column_name)
```

These methods provide efficient ways to slice and dice Pandas data, facilitating various data manipulation tasks efficiently.