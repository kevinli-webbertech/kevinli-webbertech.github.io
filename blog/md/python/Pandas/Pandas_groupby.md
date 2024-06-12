# GroupBy Operations in Pandas

Any groupby operation involves three main steps:

1. **Splitting** the object.
2. **Applying** a function.
3. **Combining** the results.

In many scenarios, we split data into sets and apply some functionality on each subset. In Pandas, the apply functionality allows us to perform various operations, including:

- **Aggregation**: Computing summary statistics.
- **Transformation**: Performing group-specific operations.
- **Filtration**: Discarding data based on some condition.

Let's dive into these operations with examples:

## Split Data into Groups

Pandas objects can be split into groups using various keys:

```python
import pandas as pd

ipl_data = {
    'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings', 'Kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
    'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
    'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
    'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]
}
df = pd.DataFrame(ipl_data)

# Group by 'Team'
grouped = df.groupby('Team')
print(grouped.groups)
```

Output:

```
{
    'Kings': Int64Index([4, 5, 6, 7], dtype='int64'),
    'Devils': Int64Index([2, 3], dtype='int64'),
    'Riders': Int64Index([0, 1, 8, 11], dtype='int64'),
    'Royals': Int64Index([9, 10], dtype='int64')
}
```

## Iterating through Groups

You can iterate through the groups like this:

```python
grouped = df.groupby('Year')

for name, group in grouped:
    print(name)
    print(group)
```

## Select a Group

You can select a single group using the `get_group()` method:

```python
grouped = df.groupby('Year')
print(grouped.get_group(2014))
```

## Aggregations

You can perform aggregation operations like sum, mean, and std:

```python
grouped = df.groupby('Year')
print(grouped['Points'].agg([np.sum, np.mean, np.std]))
```

## Transformations

Transformation applies a function to each group and returns an object indexed the same size as the group. For example:

```python
score = lambda x: (x - x.mean()) / x.std() * 10
print(grouped.transform(score))
```

## Filtration

Filtration filters the data based on a defined criteria:

```python
print(df.groupby('Team').filter(lambda x: len(x) >= 3))
```

This returns the teams that have participated three or more times in IPL.

