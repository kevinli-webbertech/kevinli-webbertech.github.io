# Python Ref- Builtin

This is compatible with v3.

## Keywords modules

In python session, there is a keyword module,

```python
>>> import keyword
>>> keyword.kwlist
['and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'with', 'yield']
```

```python
>>> keyword.iskeyword('true')
False
>>> >>> keyword.iskeyword('is')
True
```

### Get System Information

`print(help("modules"))`: print buildin module.

To master python's potential and boundaries, you only need to learn one function. That is `help()`. What happens is that it can tell you all the information about python such as builtin functions, symbols, operators...etc. Here they are,

Once you type `help()` you will enter a help session,

```python
help> 
```

You can type the following to get more info, such as `True`, `collections`, `builtins`, `modules`, `keywords`, `symbols`, `topics`, `LOOPING`.

Or in the python session you can do the same thing,

```python
>>> help('symbols')
>>> help(print)
>>> help(globals)
>>> help('builtins.globals')
>>> help('python_help_examples')
```

Then `quit` to quit the help session.

## Buildin function

Core functions

* String related

`str.lower()`, `str.upper()`, `str.capitalize()`, `str.swapcase()`, `str.title()`,`str.count( str[, beg [, end]]) `,
`islower()`, `isupper()`, `isdecimal()`, `isdigit()`, `isnumeric()`, `isalpha()`, `isalnum()`, `replace(old,new[,count])`,
`split([sep[,maxsplit]])`, `len(string)`

`find(str [,i [,j]])`: i is start, and j is end, they are optional. Returns the index of first occurence.
`index(str[,i [,j]])`: This is same as 'find' method. The only difference is that it raises the 'ValueError' exception if 'str' doesnâ€™t exist.
`rindex(str[,i [,j]])`: Searches for â€˜strâ€™ in the complete String (if i and j not defined) or in a sub-string of String (if i and j are defined). This function 			returns the last index where â€˜strâ€™ is available. If â€˜strâ€™ is not there, then it raises a ValueError exception.
`rfind(str[,i [,j]])`: This is same as find() just that this function returns the last index where 'str' is found. 
			If 'str' is not found, it returns '-1'.
`count(str[,i [,j]])`: Returns the number of occurrences of substring 'str' in the String. 
			Searches for 'str' in the complete String (if i and j not defined) or in a sub-string of String (if i and j are defined).
`join(seq)`: Returns a String obtained after concatenating the sequence â€˜seqâ€™ with a delimiter string.
`splitlines(num)`: Splits the String at line breaks and returns the list after removing the line breaks.
			Where num = if this is a positive value. It indicates that line breaks will appear in the returned list.
`lstrip([chars])`: Returns a string after removing the characters from the beginning of the String.
`rstrip()`: Returns a string after removing the characters from the End of the String.

* Formatting

`rjust(width[,fillchar])`, `ljust(width[,fillchar])`, `center(width[,fillchar])`, `zfill(width)`

* Example

`swapcase()`: flip upper and lower case.
`title()`: captilize the first letter in each word.

```python
var1='Eagle Eyes'
print (var1.count('e'))

var2='Eagle Eyes'
print (var2.count('E',0,5))
```

## Generator

## Buildin functions

### data types
bool(), float(), int(), complex(), id(), type(), str(), slice(), len()

### data structure
`set()`, `tuple()`, `frozenset()`, `dict()`, `iter()`, `next()`, `list()`, `enumerate()`, `bytearray([source[, encoding[, errors]]])`, `bytes([source[, encoding[, errors]]])`, `ord(char)`, `bin(x)`, `chr(unicode)`, `hash(object)`, `all(interable)`, `any(iterable)`, `ascii(object)`

how list and enumerate works,

```
>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>list(enumerate(seasons, start=1))
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

`bytearray([source[, encoding[, errors]]])`: source can be string, integer, object, or iterable.

`bytes([source[, encoding[, errors]]])`: Return a new â€œbytesâ€� object, which is an immutable sequence of integers in the range 0 <= x < 256.

`ord(char)`: gives the integer value of unicode.

`chr(unicode)`: gives unicode char from unicode string, reverse of ord.

`bin(x)`: converts integer to binary format.

`hash(object)`: return hash string of object

`all(iterable)`: if all elements are true, return true, otherwise false.

`any(iterable)`: if any elements are true, return true, otherwise false.

`ascii(object)`: gives a string of the object

`iter()`:

`next()`:


### Arithmetic
`min()`, `max()`, `sum()`, `abs()`, `hex()`, `oct()`, `pow()`, `round()`, `range()`, `divmod(a,b)`

### Object
super(), isinstance(), issubclass(), setattr(), getattr(),hasattr(), delattr(), filter(), breakpoint(), classmethods(), compile()

`breakpoint()`: used for debugging.
`compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)`: Use for compilation
`property()`: ?

### System and IO
help(object), print(), vars(), input(),, dir(), eval(), exec(), locals(), globals(), callable(), open(), zip(), reversed(), map(), __import__()

`globals()`:
`locals()`:

### Other
format(), all(), any()


## Ref

- https://www.tutorialspoint.com/python
- https://www.techbeamers.com/python-generator/
