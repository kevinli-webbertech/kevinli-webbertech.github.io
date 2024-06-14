# Python Ref - Quotes, comments, identifier, variable, operator

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


## Expression

lhs = rhs

rhs can be static value or expression or an existing variable.

* when rhs is an existing variable, then both will point to the same reference.

```python
>>> eval( "2.5+2.5" )
5.0
```

* multiline statement will use \ char.

```python
# Initializing a list using the multi-line statement
>>> my_list = [1, \
... 2, 3\
... ,4,5 \
... ]
>>> print(my_list)
[1, 2, 3, 4, 5]

```

* implicit line continuation
(),{},[], without completion, python assume it is multilines.

```python
>>> result = (10 + 100
... * 5 - 5
... / 100 + 10
... )
>>> print(result)
519.95

>>> subjects = [
... 'Maths',
... 'English',
... 'Science'
... ]
```

* indentation
PEP 8 suggests 4 spaces, and google suggest 2.

```python
def demo_routine(num):
  print('I am a demo function')
  if num % 2 == 0:
     return True
  else:
     return False
 
 
num = int(input('Enter a number:'))

if demo_routine(num) is True:
    print(num, 'is an even number')
else:
    print(num, 'is an odd number')
```

### Data types

* boolean
* numbers
* strings
* bytes
* Lists
* tuples
* Sets
* Dictionaries

<b>Boolean Examples</b>

`if condition` is the same as `if condition = True`

```
condition = False
if condition:
    print("You can continue with the prpgram.")
else:
    print("The program will end here.")
```

True can also serves as 1, and False is 0.

```
>>> A, B = True + 0, False + 0
>>> print(A, B)
1 0
>>> type(A), type(B)
(<class 'int'>, <class 'int'>)
```

```
>>> str = "Learn Python"
>>> len(str)
12
>>> len(str) == 12
True
>>> len(str) != 12
False
```

<b>Number has int, float and complex</b>

```
num = 2
print("The number (", num, ") is of type", type(num))

num = 3.0
print("The number (", num, ") is of type", type(num))

num = 3+5j
print("The number ", num, " is of type", type(num))
print("The number ", num, " is complex number?", isinstance(3+5j, complex))
```

complex(num1, num2) constructor will create complex number,

```python
>>> complex(1.2,5)
(1.2+5j)
```

`5 ** 2, 5 to the power of 2`

`'un' * 3, dup string 3 times`

`'abc' + 'de', string concat`

`word[:2] + word[:2], string indexing and slicing`

<b>Builtin functions for number</b>

```python
>>> num = 1234567890123456789
>>> num.bit_length()
61
```

```python
>>> import sys
>>> sys.float_info
sys.float_info(max=1.7976931348623157e+308, max_exp=1024, max_10_exp=308, min=2.2250738585072014e-308, min_exp=-1021, min_10_exp=-307, dig=15, mant_dig=53, epsilon=2.220446049250313e-16, radix=2, rounds=1)
>>> sys.float_info.dig
15
``` 


<b>String</b>
Tripple quotation mark can wrap around multiline strings.


id() function will show the memory address of the object,

```python
>>> A = 'Python3'
>>> id(A)
56272968
>>> B = A
>>> id(B)
56272968
```

Assignment will get the same address.

type() function will check data type.

```python
>>> print(type('Python String'))
<class 'str'>
>>> print(type(u'Python Unicode String'))
<class 'str'>

>>> a = 1
>>> type(a)
<type 'int'>
```

You can substring using slicing. (skipped, similar to Mathematica using square bracket)

<b>Byte</b> 

byte object is 8 bits, so it is from 0-255. 

TODO

https://www.techbeamers.com/python-data-types-learn-basic-advanced/


**List**

* Lists in Python can be declared by placing elements inside square brackets separated by commas.
* Elements can be any type.
* Index from 0.
* Supports slicing.

**Tuple**
* A tuple is a heterogeneous collection of Python objects separated by commas.
* Both objects are an ordered sequence.
* They enable indexing and repetition and index from 0.
* Nesting is allowed.
* Support slicing, using square bracket.
* Tuples do differ a bit from the list as they are immutable.

eg:

`# How does repetition work with tuples
sample_tuple = ('Python 3',)*3
print(sample_tuple)
`

**Sets**

* Use {}, unorder and immutable.
* Set is optimized for checking if it contains an element or not, faster than list.

Two ways to create set, one is to use set() function, 
and the other is to use {},

```python
>>> sample_set = set("Python data types")
>>> type(sample_set)
<class 'set'>
>>> sample_set
{'e', 'y', 't', 'o', ' ', 'd', 's', 'P', 'p', 'n', 'h', 'a'}
```

another way,

```python
>>> another_set = {'red', 'green', 'black'}
>>> type(another_set)
<class 'set'>
>>> another_set
{'red', 'green', 'black'}

```

**Frozen set**

Frozen set is immutable,

```Python
>>> sample_set = {"red", "green"}
>>> sample_set
{'green', 'red'}
>>> print(sample_set)
{'green', 'red'}
>>> sample_set.add("black")
>>> sample_set
{'green', 'black', 'red'}
>>> frozen_set = frozenset(["red", "green", "black"])
>>> frozen_set.add('hello')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'frozenset' object has no attribute 'add'
```

**Dictionaries**

* Python syntax for creating dictionaries use braces {} where each item appears as a pair of keys and values. The key and value can be of any Python data types.

***Mutable***

```python
>>> sample_dict = {'key':'value', 'jan':31, 'feb':28, 'mar':31}
>>> type(sample_dict)
<class 'dict'>
>>> sample_dict
{'mar': 31, 'key': 'value', 'jan': 31, 'feb': 28}

```

Reference it value with key,

```python
>>> sample_dict['jan']
31
>>> sample_dict['feb']
28
```

Dictionary methods,

* keys() It isolates the keys from a dictionary.
* values() It isolates the values from a dictionary.
* items() It returns the items in a list style of (key, value) pairs.

```python
>>> sample_dict.keys()
dict_keys(['mar', 'key', 'jan', 'feb'])
>>> sample_dict.values()
dict_values([31, 'value', 31, 28])
>>> sample_dict.items()
dict_items([('mar', 31), ('key', 'value'), ('jan', 31), ('feb', 28)])
```

Modify dictionaries

```python
>>> sample_dict['feb'] = 29
>>> sample_dict
{'mar': 31, 'key': 'value', 'jan': 31, 'feb': 29}
>>> sample_dict.update({'apr':30})
>>> sample_dict
{'apr': 30, 'mar': 31, 'key': 'value', 'jan': 31, 'feb': 29}
>>> del sample_dict['key']
>>> sample_dict
{'apr': 30, 'mar': 31, 'jan': 31, 'feb': 29}
```

### ref
- https://www.techbeamers.com/python-format-string-list-dict/

- https://www.techbeamers.com/python-operators-tutorial-beginners/

- https://www.techbeamers.com/python-operator-precedence-associativity/
