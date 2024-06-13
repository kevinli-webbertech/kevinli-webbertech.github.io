# Categorical Data in Pandas

Categorical variables are a Pandas data type used for variables with a limited, usually fixed number of possible values. They are useful for optimizing memory usage and defining logical orderings.

## Object Creation

### `category`

```python
import pandas as pd

s = pd.Series(["a", "b", "c", "a"], dtype="category")
print(s)
```

Output:

```
0    a
1    b
2    c
3    a
dtype: category
Categories (3, object): [a, b, c]
```

### `pd.Categorical`

```python
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
print(cat)
```

Output:

```
[a, b, c, a, b, c]
Categories (3, object): [a, b, c]
```

### Description

```python
cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
df = pd.DataFrame({"cat": cat, "s": ["a", "c", "c", np.nan]})

print(df.describe())
print(df["cat"].describe())
```

### Get Category Properties

```python
s = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
print(s.categories)
print(s.ordered)
```

## Renaming Categories

```python
s = pd.Series(["a","b","c","a"], dtype="category")
s.cat.categories = ["Group %s" % g for g in s.cat.categories]
print(s.cat.categories)
```

## Appending New Categories

```python
s = pd.Series(["a","b","c","a"], dtype="category")
s = s.cat.add_categories([4])
print(s.cat.categories)
```

## Removing Categories

```python
s = pd.Series(["a","b","c","a"], dtype="category")
print(s.cat.remove_categories("a"))
```

## Comparison of Categorical Data

```python
cat = pd.Series([1,2,3]).astype("category", categories=[1,2,3], ordered=True)
cat1 = pd.Series([2,2,2]).astype("category", categories=[1,2,3], ordered=True)

print(cat > cat1)
```

---

These examples illustrate the creation, manipulation, and comparison of categorical data in Pandas. 