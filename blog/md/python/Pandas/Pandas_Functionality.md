# Pandas Data Structures and Basic Functionality

## Overview

Pandas is a powerful library in Python for data manipulation and analysis. It provides three main data structures:

1. **Series**: A one-dimensional labeled array capable of holding any data type.
2. **DataFrame**: A two-dimensional labeled data structure with columns of potentially different types.
3. **Panel**: A three-dimensional container of data.

### Panel Data

The Panel is a three-dimensional data structure that can be created in several ways:

- From ndarrays
- From a dict of DataFrames
- From 3D ndarray

#### Example: Creating a Panel

```python
import pandas as pd
import numpy as np

# Creating a Panel from a 3D ndarray
data = np.random.rand(2, 4, 5)
p = pd.Panel(data)
print(p)
```

Output:

```
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 4 (major_axis) x 5 (minor_axis)
Items axis: 0 to 1
Major_axis axis: 0 to 3
Minor_axis axis: 0 to 4
```

### Series Basic Functionality

A Series is a one-dimensional array-like object that can hold any data type. It has several attributes and methods for data manipulation.

#### Attributes and Methods

1. **axes**: Returns a list of the row axis labels.
2. **dtype**: Returns the dtype of the object.
3. **empty**: Returns True if the series is empty.
4. **ndim**: Returns the number of dimensions (1 for Series).
5. **size**: Returns the number of elements.
6. **values**: Returns the Series as an ndarray.
7. **head()**: Returns the first n rows.
8. **tail()**: Returns the last n rows.

#### Example: Series Attributes and Methods

```python
import pandas as pd
import numpy as np

# Create a Series with random numbers
s = pd.Series(np.random.randn(4))
print(s)

# Axes
print("Axes:", s.axes)

# Empty
print("Is empty?", s.empty)

# Number of dimensions
print("Dimensions:", s.ndim)

# Size
print("Size:", s.size)

# Values
print("Values:", s.values)

# Head
print("First two rows:\n", s.head(2))

# Tail
print("Last two rows:\n", s.tail(2))
```

### DataFrame Basic Functionality

A DataFrame is a two-dimensional labeled data structure with columns of potentially different types.

#### Attributes and Methods

1. **T**: Transposes rows and columns.
2. **axes**: Returns a list with the row axis labels and column axis labels.
3. **dtypes**: Returns the data types of each column.
4. **empty**: Returns True if the DataFrame is empty.
5. **ndim**: Number of axes / array dimensions.
6. **shape**: Returns a tuple representing the dimensionality.
7. **size**: Number of elements in the DataFrame.
8. **values**: Numpy representation of DataFrame.
9. **head()**: Returns the first n rows.
10. **tail()**: Returns the last n rows.

#### Example: DataFrame Attributes and Methods

```python
import pandas as pd
import numpy as np

# Create a dictionary of series
data = {
    'Name': pd.Series(['Alice', 'Bob', 'Charlie', 'David', 'Eve']),
    'Age': pd.Series([24, 30, 22, 25, 28]),
    'Rating': pd.Series([4.5, 3.7, 4.0, 3.8, 4.2])
}

# Create a DataFrame
df = pd.DataFrame(data)
print("DataFrame:\n", df)

# Transpose
print("Transpose:\n", df.T)

# Axes
print("Axes:", df.axes)

# Data types
print("Data types:\n", df.dtypes)

# Is empty?
print("Is empty?", df.empty)

# Number of dimensions
print("Dimensions:", df.ndim)

# Shape
print("Shape:", df.shape)

# Size
print("Size:", df.size)

# Values
print("Values:\n", df.values)

# Head
print("First two rows:\n", df.head(2))

# Tail
print("Last two rows:\n", df.tail(2))
```

These examples and descriptions provide a basic understanding of how to use and manipulate Series and DataFrame objects in Pandas. For more advanced operations and detailed documentation, refer to the [Pandas documentation](https://pandas.pydata.org/docs/).