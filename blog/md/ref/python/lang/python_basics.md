# Python Basics -- Quotes, comments, identifier, variable, operator

## Quotes

* Quotes

1/ single quote, double quote, tripple quote are both the same if quote around one line of string

2/ For multiline, use triple quotes

```python
>>> str = """A multiline string
starts and ends with
a triple quotation mark."""
>>> str
'A multiline string\nstarts and ends with\na triple quotation mark.'
```

3/ print() function will print \n a return.

## comments

use `#` for single line or multiple line

## Identifier

* a-z, A-Z, 0-9, _, can be mix-used to create identifier except that 0-9 can not be the first letter.

* special chars['.', '!', '@', '#', '$', '%'] can not be used to create identifier.

* keyword module provides function to test keyword

or check if it is an identifier,

```python
>>> 'techbeamers'.isidentifier()
True
>>> '1techbeamers'.isidentifier()
False
>>> 'techbeamers.com'.isidentifier()
False
>>> 'techbemaers_com'.isidentifier()
True
```

* Class name should start with capital letter, other var should start with lowercase.
* private identifier should start with _.
* Don't use _ as leading and trailing char as buildin vars do that.

## Variable

For each variable, it is an object that holds the value, you can see the reference by doing the following. If you reassign, then the old one goes to garbage collector.

```
>>> test=10
>>> id(test)
140619828706608
>>> test=11
>>> id(test)
140619828706584
```

* For optimization, python builds a cache and reuses some of the immutable integers and strings.

* Object is a region of memory of the following,

1/ hold object value
2/ hold a counter for garbage collector
3/ A type designator to reflect object type

* Check object type by using type() function

```python
>>> test = 10
>>> type(test)
<class 'int'>
>>> test = 'techbeamers'
>>> type(test)
<class 'str'>
>>> test = {'Python', 'C', 'C++'}
>>> type(test)
<class 'set'>
>>>
```

### Operator

Most operators are similar to other languages, for different ones, they are listed here,

* `+`, string concates, eg: `"hello"+ "world"`
* `+=`, if both sides are data structure, then it means adds elements into the one on the left.
* `*`, string repetition, eg: `"hello"*3`
* `[]`, slicing, similar to Wolfram language, [:n], [n:], [n:m], m can be negative.
* `in`, `not in`,  existing or not existing, 

eg:

```python

var1 = 'Python'
print ('n' in var1)
 True
```

``` python

var1 = 'Python'
print ('N' not in var1)
 True
```

* `for`, eg: `for var in var1: print (var, end ="")`
* r/R, raw string, eg: `print (r'\n')`
* %, formatting, eg: `print ("Employee Name: %s,\nEmployee Age:%d" % ('Ashish',25))`, pay attention to the last one.
* \, escape char
* u, unicode character will be treated as 16 bit. eg: `print (u' Hello Python!!')`