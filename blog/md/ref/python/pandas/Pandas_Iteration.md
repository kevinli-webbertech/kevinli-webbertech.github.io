# Iterating over Pandas Objects

When iterating over Pandas objects like Series, DataFrame, or Panel, the behavior depends on the type of object being iterated.

## Basic Iteration

Basic iteration (using `for` loop) produces the following:

- **Series**: Values
- **DataFrame**: Column labels
- **Panel**: Item labels

### Iterating a DataFrame

Iterating over a DataFrame yields column names.

#### Example

```python
import pandas as pd
import numpy as np

N = 20
df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01', periods=N, freq='D'),
   'x': np.linspace(0, stop=N-1, num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low', 'Medium', 'High'], N).tolist(),
   'D': np.random.normal(100, 10, size=N).tolist()
   })

for col in df:
   print(col)
```

Output:
```
A
C
D
x
y
```

### Iterating over Rows

To iterate over the rows of a DataFrame, you can use various functions:

- `iteritems()`: Iterate over (key, value) pairs
- `iterrows()`: Iterate over rows as (index, series) pairs
- `itertuples()`: Iterate over rows as namedtuples

#### `iteritems()`

Iterates over each column as a key-value pair, with the label as the key and the column value as a Series object.

#### Example

```python
import pandas as pd
import numpy as np
 
df = pd.DataFrame(np.random.randn(4, 3), columns=['col1', 'col2', 'col3'])
for key, value in df.iteritems():
   print(key, value)
```

Output:
```
col1 0   -0.019812
1   -0.329456
2   -1.431924
3   -0.291681
Name: col1, dtype: float64

col2 0   -0.240592
1   -0.624822
2    1.083839
3   -0.457297
Name: col2, dtype: float64

col3 0    0.961316
1    0.157554
2   -0.734729
3   -0.182616
Name: col3, dtype: float64
```

#### `iterrows()`

`iterrows()` returns the iterator yielding each index value along with a series containing the data in each row.

#### Example

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(4, 3), columns=['col1', 'col2', 'col3'])
for row_index, row in df.iterrows():
   print(row_index, row)
```

Output:
```
0 col1   -0.017876
col2    0.195776
col3    1.374835
Name: 0, dtype: float64

1 col1   -0.520765
col2    0.014285
col3   -0.413532
Name: 1, dtype: float64

2 col1   -0.617154
col2    0.167004
col3   -1.320788
Name: 2, dtype: float64

3 col1   -0.276741
col2    1.055341
col3    1.051195
Name: 3, dtype: float64
```

#### `itertuples()`

`itertuples()` method returns an iterator yielding a named tuple for each row in the DataFrame.

#### Example

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(4, 3), columns=['col1', 'col2', 'col3'])
for row in df.itertuples():
    print(row)
```

Output:
```
Pandas(Index=0, col1=-0.7013021283838773, col2=-0.49808805649489635, col3=0.1536626230329705)
Pandas(Index=1, col1=-0.17610431619653718, col2=-0.45019607188571925, col3=-0.8717462898927126)
Pandas(Index=2, col1=0.4240626498142766, col2=0.8589760730912736, col3=-1.2424764475846963)
Pandas(Index=3, col1=0.4777894675178841, col2=0.013717831289479929, col3=0.4692152760150211)
```

**Note**: Avoid modifying objects while iterating as it returns a copy of the original object, and changes won't reflect on the original object.