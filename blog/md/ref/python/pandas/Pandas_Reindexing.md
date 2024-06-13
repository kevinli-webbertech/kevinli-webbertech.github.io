# Reindexing and Renaming in Pandas

## Reindexing

Reindexing changes the row labels and column labels of a DataFrame to conform to a given set of labels along a particular axis. Multiple operations can be accomplished through reindexing, such as:

- Reordering the existing data to match a new set of labels.
- Inserting missing value (NA) markers in label locations where no data for the label existed.

### Example

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

# Reindex the DataFrame
df_reindexed = df.reindex(index=[0, 2, 5], columns=['A', 'C', 'B'])

print(df_reindexed)
```

Output:
```
            A    C    B
0  2016-01-01  Low  NaN
2  2016-01-03  High NaN
5  2016-01-06  Low  NaN
```

### Reindex to Align with Other Objects

You can reindex one DataFrame to align with another object.

#### Example

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(10, 3), columns=['col1', 'col2', 'col3'])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=['col1', 'col2', 'col3'])

df1 = df1.reindex_like(df2)
print(df1)
```

Output:
```
         col1      col2      col3
0  -2.467652  -1.211687  -0.391761
1  -0.287396   0.522350   0.562512
2  -0.255409  -0.483250   1.866258
3  -1.150467  -0.646493  -0.222462
4   0.152768  -2.056643   1.877233
5  -1.155997   1.528719  -1.343719
6  -1.015606  -1.245936  -0.295275
```

### Filling while Reindexing

The `reindex()` method has an optional `method` parameter to fill missing values:
- `pad`/`ffill` - Fill values forward
- `bfill`/`backfill` - Fill values backward
- `nearest` - Fill from the nearest index values

#### Example

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(6, 3), columns=['col1', 'col2', 'col3'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['col1', 'col2', 'col3'])

# Padding NAN's
print(df2.reindex_like(df1))

# Now Fill the NAN's with preceding Values
print("Data Frame with Forward Fill:")
print(df2.reindex_like(df1, method='ffill'))
```

Output:
```
         col1      col2      col3
0   1.311620 -0.707176  0.599863
1  -0.423455 -0.700265  1.133371
2        NaN       NaN       NaN
3        NaN       NaN       NaN
4        NaN       NaN       NaN
5        NaN       NaN       NaN

Data Frame with Forward Fill:
         col1      col2      col3
0   1.311620 -0.707176  0.599863
1  -0.423455 -0.700265  1.133371
2  -0.423455 -0.700265  1.133371
3  -0.423455 -0.700265  1.133371
4  -0.423455 -0.700265  1.133371
5  -0.423455 -0.700265  1.133371
```

### Limits on Filling while Reindexing

The `limit` argument provides control over filling by specifying the maximum count of consecutive matches.

#### Example

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(6, 3), columns=['col1', 'col2', 'col3'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['col1', 'col2', 'col3'])

# Padding NAN's
print(df2.reindex_like(df1))

# Now Fill the NAN's with preceding Values
print("Data Frame with Forward Fill limiting to 1:")
print(df2.reindex_like(df1, method='ffill', limit=1))
```

Output:
```
         col1      col2      col3
0   0.247784  2.128727  0.702576
1  -0.055713 -0.021732 -0.174577
2        NaN       NaN       NaN
3        NaN       NaN       NaN
4        NaN       NaN       NaN
5        NaN       NaN       NaN

Data Frame with Forward Fill limiting to 1:
         col1      col2      col3
0   0.247784  2.128727  0.702576
1  -0.055713 -0.021732 -0.174577
2  -0.055713 -0.021732 -0.174577
3        NaN       NaN       NaN
4        NaN       NaN       NaN
5        NaN       NaN       NaN
```

## Renaming

The `rename()` method allows relabeling of an axis based on some mapping (a dictionary or Series) or an arbitrary function.

### Example

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(6, 3), columns=['col1', 'col2', 'col3'])
print(df1)

print("After renaming the rows and columns:")
print(df1.rename(columns={'col1': 'c1', 'col2': 'c2'}, index={0: 'apple', 1: 'banana', 2: 'durian'}))
```

Output:
```
         col1      col2      col3
0   0.486791  0.105759  1.540122
1  -0.990237  1.007885 -0.217896
2  -0.483855 -1.645027 -1.194113
3  -0.122316  0.566277 -0.366028
4  -0.231524 -0.721172 -0.112007
5   0.438810  0.000225  0.435479

After renaming the rows and columns:
                c1        c2      col3
apple     0.486791  0.105759  1.540122
banana   -0.990237  1.007885 -0.217896
durian   -0.483855 -1.645027 -1.194113
3        -0.122316  0.566277 -0.366028
4        -0.231524 -0.721172 -0.112007
5         0.438810  0.000225  0.435479
```

### Inplace Renaming

The `rename()` method provides an `inplace` parameter, which by default is `False` and creates a new DataFrame. Use `inplace=True` to rename the data in place.

Example with `inplace` parameter:
```python
df1.rename(columns={'col1': 'c1', 'col2': 'c2'}, index={0: 'apple', 1: 'banana', 2: 'durian'}, inplace=True)
print(df1)
```

This will modify `df1` directly with the new names.