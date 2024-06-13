# String Operations in Pandas

We will explore various string operations that can be performed on basic Series/Index in Pandas. These operations are essential for text data preprocessing and manipulation. Let's dive into each operation along with examples.

## 1. `lower()`

Converts strings in the Series/Index to lower case.

```python
import pandas as pd

data = pd.Series(['Hello', 'WORLD', 'pandas'])
data.str.lower()
```

## 2. `upper()`

Converts strings in the Series/Index to upper case.

```python
data.str.upper()
```

## 3. `len()`

Computes the length of each string.

```python
data.str.len()
```

## 4. `strip()`

Strips whitespace from each string in the Series/Index from both sides.

```python
data = pd.Series(['   apple  ', 'banana  ', ' orange'])
data.str.strip()
```

## 5. `split(' ')`

Splits each string with the given pattern.

```python
data.str.split(' ')
```

## 6. `cat(sep=' ')`

Concatenates the series/index elements with the given separator.

```python
data.str.cat(sep=', ')
```

## 7. `get_dummies()`

Returns the DataFrame with One-Hot Encoded values.

```python
pd.get_dummies(data)
```

## 8. `contains(pattern)`

Returns a Boolean value True for each element if the substring contains in the element, else False.

```python
data.str.contains('na')
```

## 9. `replace(a, b)`

Replaces the value a with the value b.

```python
data.str.replace('a', 'X')
```

## 10. `repeat(value)`

Repeats each element with the specified number of times.

```python
data.str.repeat(2)
```

## 11. `count(pattern)`

Returns the count of appearances of the pattern in each element.

```python
data.str.count('a')
```

## 12. `startswith(pattern)`

Returns true if the element in the Series/Index starts with the pattern.

```python
data.str.startswith('a')
```

## 13. `endswith(pattern)`

Returns true if the element in the Series/Index ends with the pattern.

```python
data.str.endswith('e')
```

## 14. `find(pattern)`

Returns the first position of the first occurrence of the pattern.

```python
data.str.find('e')
```

## 15. `findall(pattern)`

Returns a list of all occurrences of the pattern.

```python
data.str.findall('a')
```

## 16. `swapcase()`

Swaps the case lower/upper.

```python
data.str.swapcase()
```

## 17. `islower()`

Checks whether all characters in each string in the Series/Index are lowercase or not. Returns Boolean.

```python
data.str.islower()
```

## 18. `isupper()`

Checks whether all characters in each string in the Series/Index are uppercase or not. Returns Boolean.

```python
data.str.isupper()
```

## 19. `isnumeric()`

Checks whether all characters in each string in the Series/Index are numeric. Returns Boolean.

```python
data.str.isnumeric()
```

These string operations in Pandas provide a powerful toolkit for handling and manipulating text data efficiently within your data analysis workflow.