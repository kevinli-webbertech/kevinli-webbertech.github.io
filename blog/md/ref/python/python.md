# Python Cheatsheet

This is compatible with v3.

## Basics

### Quotes, comments

* Quotes

1/ single quote, double quote, tripple quote are both the same if quote around one line of string

2/ For multiline, use triple quotes

```
>>> str = """A multiline string
starts and ends with
a triple quotation mark."""
>>> str
'A multiline string\nstarts and ends with\na triple quotation mark.'
```

3/ print() function will print \n a return.

* comments
use `#` for single line or multiple line

### Keywords modules

In python session, there is a keyword module,

```
>>> import keyword
>>> keyword.kwlist
['and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'with', 'yield']
```
q
```
>>> keyword.iskeyword('true')
False
>>> >>> keyword.iskeyword('is')
True
```

### Deep dive to get system information

`print(help("modules"))`: print buildin module.

To master python's potential and boundaries, you only need to learn one function. That is `help()`. What happens is that it can tell you all the information about python such as builtin functions, symbols, operators...etc. Here they are,

Once you type `help()` you will enter a help session,

```
help> 
```

You can type the following to get more info, such as `True`, `collections`, `builtins`, `modules`, `keywords`, `symbols`, `topics`, `LOOPING`.

Or in the python session you can do the same thing,


```
>>> help('symbols')
>>> help(print)
>>> help(globals)
>>> help('builtins.globals')
>>> help('python_help_examples')
```

Then `quit` to quit the help session.


### Identifier
* a-z, A-Z, 0-9, _, can be mix-used to create identifier except that 0-9 can not be the first letter.

* special chars['.', '!', '@', '#', '$', '%'] can not be used to create identifier.

* keyword module provides function to test keyword\



or check if it is an identifier,

```
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

### Variable

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

```
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

```
# example
var1 = 'Python'
print ('n' in var1)
 True
```

```
# example
var1 = 'Python'
print ('N' not in var1)
 True
```

* `for`, eg: `for var in var1: print (var, end ="")`
* r/R, raw string, eg: `print (r'\n')`
* %, formatting, eg: `print ("Employee Name: %s,\nEmployee Age:%d" % ('Ashish',25))`, pay attention to the last one.
* \, escape char
* u, unicode character will be treated as 16 bit. eg: `print (u' Hello Python!!')`

### Buildin function

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

```
var1='Eagle Eyes'
print (var1.count('e'))

var2='Eagle Eyes'
print (var2.count('E',0,5))
```

### Expression

lhs = rhs

rhs can be static value or expression or an existing variable.

* when rhs is an existing variable, then both will point to the same reference.

```
>>> eval( "2.5+2.5" )
5.0
```

* multiline statement will use \ char.

```
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

```
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

```
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

```
>>> complex(1.2,5)
(1.2+5j)
```

`5 ** 2, 5 to the power of 2`

`'un' * 3, dup string 3 times`

`'abc' + 'de', string concat`

`word[:2] + word[:2], string indexing and slicing`

<b>Builtin functions for number</b>

```
>>> num = 1234567890123456789
>>> num.bit_length()
61
```



```
>>> import sys
>>> sys.float_info
sys.float_info(max=1.7976931348623157e+308, max_exp=1024, max_10_exp=308, min=2.2250738585072014e-308, min_exp=-1021, min_10_exp=-307, dig=15, mant_dig=53, epsilon=2.220446049250313e-16, radix=2, rounds=1)
>>> sys.float_info.dig
15
``` 

<b>String</b>
Tripple quotation mark can wrap around multiline strings.



id() function will show the memory address of the object,

```
>>> A = 'Python3'
>>> id(A)
56272968
>>> B = A
>>> id(B)
56272968
```

Assignment will get the same address.

type() function will check data type.

```
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

<b>List</b>

* Lists in Python can be declared by placing elements inside square brackets separated by commas.
* Elements can be any type.
* Index from 0.
* Supports slicing.

<b>Tuple</b>
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

<b> Sets </b>
* Use {}, unorder and immutable.
* Set is optimized for checking if it contains an element or not, faster than list.

Two ways to create set, one is to use set() function, 
and the other is to use {},

```
>>> sample_set = set("Python data types")
>>> type(sample_set)
<class 'set'>
>>> sample_set
{'e', 'y', 't', 'o', ' ', 'd', 's', 'P', 'p', 'n', 'h', 'a'}

```

another way,

```
>>> another_set = {'red', 'green', 'black'}
>>> type(another_set)
<class 'set'>
>>> another_set
{'red', 'green', 'black'}

```

Frozen set

Frozen set is immutable,

```
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

<b>dictionaries</b>

* Python syntax for creating dictionaries use braces {} where each item appears as a pair of keys and values. The key and value can be of any Python data types.

* Mutable

```

>>> sample_dict = {'key':'value', 'jan':31, 'feb':28, 'mar':31}
>>> type(sample_dict)
<class 'dict'>
>>> sample_dict
{'mar': 31, 'key': 'value', 'jan': 31, 'feb': 28}

```

Reference it value with key,

```
>>> sample_dict['jan']
31
>>> sample_dict['feb']
28
```

Dictionary methods,

* keys() â€“ It isolates the keys from a dictionary.
* values() â€“ It isolates the values from a dictionary.
* items() â€“ It returns the items in a list style of (key, value) pairs.

```
>>> sample_dict.keys()
dict_keys(['mar', 'key', 'jan', 'feb'])
>>> sample_dict.values()
dict_values([31, 'value', 31, 28])
>>> sample_dict.items()
dict_items([('mar', 31), ('key', 'value'), ('jan', 31), ('feb', 28)])
```

Modify dictionaries

```
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


https://www.techbeamers.com/python-format-string-list-dict/

https://www.techbeamers.com/python-operators-tutorial-beginners/

https://www.techbeamers.com/python-operator-precedence-associativity/

## if/else

<b>if and io operation </b>

```
days = int(input("How many days in a leap year? "))
if days == 366:
    print("You have cleared the test.")
print("Congrats!")
```

<b>if/else </b>

```
answer = input("Is Python an interpreted language? Yes or No >> ").lower()

if answer == "yes" :
    print("You have cleared the test.")
else :
    print("You have failed the test.")

print("Thanks!")
```

<b>if/else on one line </b>

If Else in one line - Syntax

`value_on_true if condition else value_on_false`


Example,

```
>>> num = 2
>>> 'Even' if num%2 == 0 else 'Odd'
'Even'
>>> num = 3
>>> 'Even' if num%2 == 0 else 'Odd'
'Odd'
>>> num = 33
>>> 'Even' if num%2 == 0 else 'Odd'
'Odd'
>>> num = 34
>>> 'Even' if num%2 == 0 else 'Odd'
'Even'
>>>

```

<b>if/elif/else </b>

```
while True:
    response = input("Which Python data type is an ordered sequence? ").lower()
    print("You entered:", response)
    
    if response == "list" :
        print("You have cleared the test.")
        break
    elif response == "tuple" :
        print("You have cleared the test.")
        break
    else :
        print("Your input is wrong. Please try again.")
        
```


<b> nested if example </b>

```
x = 10
y = 20
z = 30

print("Start")
if x == 10:
    print(" Nested If")
    if y == 20:
        print(" End of Nested If Block ")
    else:
        print(" End of Nested If-Else Block ")
elif y == 20:
    print(" Elif block ")
else:
    print(" Nested If")
    if z == 30:
        print(" End of Nested If Block ")
    else:
        print(" End of Nested If-Else Block ")
print("Stop")

```

<b>Using Not Operator With Python If Else</b>

```
a = 10
b = 20
if not a > b :
    print("The number %d is less than %d" %(a, b))/
```

## Loops

```
for iter in sequence:
    statements(iter)
```

Example,

```
vowels="AEIOU"
for iter in vowels:
    print("char:", iter)
```     

Another example,

```
int_list = [1, 2, 3, 4, 5, 6]
sum = 0
for iter in int_list:
    sum += iter
print("Sum =", sum)
print("Avg =", sum/len(int_list))
```

<b>Range() function</b>
A similar function is also seen in Mathematica.

Example,

```
for iter in range(0, 3):
    print("iter: %d" % (iter))
```

Another example

```
>>> books = ['C', 'C++', 'Java', 'Python']
>>> for index in range(len(books)):
...     print('Book (%d):' , index, books[index])
```

https://www.techbeamers.com/python-while-loop/
https://www.techbeamers.com/python-switch-case-statement/

<b>Else Clause With Python For Loop </b>
Interestingly, Python allows using an optional else statement along with the â€œforâ€� loop.

The code under the else clause executes after the completion of the â€œforâ€� loop. However, if the loop stops due to a â€œbreakâ€� call, then itâ€™ll skip the â€œelseâ€� clause.

Foe-Else Syntax

```

for item in seq:
    statement 1
    statement 2
    if <cond>:
        break
else:
    statements
    
```

## Iterators

* In Python, an iterator is an object which implements the iterator protocol, which consist of the methods __iter__() and __next__().
* Lists, tuples, dictionaries, and sets are all iterable objects. They are iterable containers which you can get an iterator from.
* All these objects have a iter() method which is used to get an iterator.

```
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))
```

Same thing for string mystr = "banana", all the above can be done by for loop,

```
mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)
```

and

```
mystr = "banana"

for x in mystr:
  print(x)
```

## Name and Scope
Scope are similar to Java and C, however, there is a keyword `global` where inside of a function, you can declare a variable is global,
or if you want to change the global var's value, you can claim it is global.

```
def myfunc():
  global x
  x = 300

myfunc()

print(x)

```

```
x = 300

def myfunc():
  global x
  x = 200

myfunc()

print(x)
```

https://www.techbeamers.com/python-namespace-scope/


## Function and class

### function

```
def my_function():
  print("Hello from a function")
```

revoke function

`my_function()`

### class

```
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
    
  def myfunc(self):
    print("Hello my name is " + self.name)
```

init class and use class,

```
p1 = Person("John", 36)
print(p1.name)
print(p1.age)
p1.myfunc()
```

The above `self` does not need to be self, you can call it `this` or anything such as `mysillyobject`.

### Delete a class property

Delete the age property from the p1 object:

`del p1.age`

### Delete an object

`del p1`

### pass statement

class definition can not be null, but `pass` can avoid the error.

`class Person:
 pass
`

### inheritance, super() function

The idea is similar to Java. In child class, the __init__() class is constructor that overrides the parent constructor, thus requires using `super()`
to call the parent class's constructor.

```
class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)
    
class Student(Person):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year
  
  def welcome(self):
    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)
    
    
x = Student("Mike", "Olsen", 2019)

```

In above example, child class add a new property as well, in the super() you can also call `Person` to replace it.

https://www.techbeamers.com/python-function/
https://www.techbeamers.com/python-class/
https://www.techbeamers.com/python-inheritance/

## Lambda

Lambda function allows anonymous function definition with unlimited number of arguments and allow function to return a lambda function.

```
x = lambda a : a + 10
print(x(5))

x = lambda a, b : a * b
print(x(5, 6))

x = lambda a, b, c : a + b + c
print(x(5, 6, 2))

def myfunc(n):
  return lambda a : a * n

mytripler = myfunc(3)
print(mytripler(11))
```
https://www.techbeamers.com/python-lambda/

## Module

Module is the concept of library, it deals with `from`, `import`. Normally a library has extension of `.py`.
`from xxx import yyy` is similar to Java's static import. However, you can not write `import yyy from xxx`.

mymodule.py

```
def greeting(name):
  print("Hello, " + name)
```

hello-world.py

```
import mymodule
def my_function(name):
    mymodule.greeting(name)

my_function("xiaofeng")
```

```
from mymodule import greeting
def my_function(name):
    greeting(name)

my_function("xiaofeng")
```

https://www.techbeamers.com/python-module/

## Exception and IO

### Assert

`assert Expression[, Arguments]`

```
#!/usr/bin/python
def KelvinToFahrenheit(Temperature):
   assert (Temperature >= 0),"Colder than absolute zero!"
   return ((Temperature-273)*1.8)+32
print KelvinToFahrenheit(273)
print int(KelvinToFahrenheit(505.78))
print KelvinToFahrenheit(-5)
```

### Exception

* try catch, finally paradigm, but a little different, `else` is not `finally`

```
try:
   You do your operations here;
   ......................
except ExceptionI:
   If there is ExceptionI, then execute this block.
except ExceptionII:
   If there is ExceptionII, then execute this block.
   ......................
else:
   If there is no exception then execute this block. 
```

Example 1

```
#!/usr/bin/python

try:
   fh = open("testfile", "r")
   fh.write("This is my test file for exception handling!!")
except IOError:
   print "Error: can\'t find file or read data"
else:
   print "Written content in the file successfully"
   fh.close()
```

* The except Clause with No Exceptions

```
try:
   You do your operations here;
   ......................
except:
   If there is any exception, then execute this block.
   ......................
else:
   If there is no exception then execute this block. 
```


* The except Clause with Multiple Exceptions

```
try:
   You do your operations here;
   ......................
except(Exception1[, Exception2[,...ExceptionN]]]):
   If there is any exception from the given exception list, 
   then execute this block.
   ......................
else:
   If there is no exception then execute this block.
```

* The try-finally Clause

This is similar to Java,

```
try:
   You do your operations here;
   ......................
   Due to any exception, this may be skipped.
finally:
   This would always be executed.
   ......................
```

Example

```
#!/usr/bin/python

try:
   fh = open("testfile", "w")
   try:
      fh.write("This is my test file for exception handling!!")
   finally:
      print "Going to close the file"
      fh.close()
except IOError:
   print "Error: can\'t find file or read data"
```

* Argument of Exception

```
#!/usr/bin/python

# Define a function here.
def temp_convert(var):
   try:
      return int(var)
   except ValueError, Argument:
      print "The argument does not contain numbers\n", Argument

# Call above function here.
temp_convert("xyz");
```

* Raising Exception

`raise [Exception [, args [, traceback]]]`

```
def functionName( level ):
   if level < 1:
      raise "Invalid level!", level
      # The code below to this would not be executed
      # if we raise the exception

try:
   Business Logic here...
except "Invalid level!":
   Exception handling here...
else:
   Rest of the code here...
   
```

* User defined exception

Newer python requires user defined exeption

Example,

```
#!/usr/bin/python

class UserDefinedError(RuntimeError):
   def __init__(self, arg):
      self.args = arg
      
def functionName(level):
   if level < 1:
      raise UserDefinedError("this is my exception"), level
      # The code below to this would not be executed
      # if we raise the exception
      
try:
   functionName(10)
except UserDefinedError,e:
   print(e.args)
else:
   print("Rest of the code here...")
   
try:
   functionName(0)
except UserDefinedError,e:
   print(e.args)
else:
   print("Rest of the code here...")
```
https://www.techbeamers.com/python-copy-file/
https://www.techbeamers.com/use-try-except-python/


## IO

* raw_input(prompt), input(prompt)

raw_input: input is treated as string from keyboard.
input: input is treated as valid python expression and will be evaluated.

* Opening and Closing Files

`file object = open(file_name [, access_mode][, buffering])`

`fileObject.write(string)`

`fileObject.read([count])`

`fileObject.close()`

* file attributes

file.closed (boolean), file.mode(access mode), file.name, file.softspace

* file positions

`fileObject.tell()`: tells the current position within the file. 

`fileObject.seek(offset[,from])`: 

The `offset` argument indicates the number of bytes to be moved. 
`from` means, If from is set to 0, it means use the beginning of the file as the reference position and 1 means use the current position as the reference position and if it is set to 2 then the end of the file would be taken as the reference position.


Example

```
#!/usr/bin/python

# Write to a file
fo = open("foo.txt", "wb")
fo.write( "Python is a great language.\nYeah its great!!\n")

# Open a file
fo = open("foo.txt", "r+")
str = fo.read(10)
print "Read String is : ", str

# Check current position
position = fo.tell()
print "Current file position : ", position

# Reposition pointer at the beginning once again
position = fo.seek(0, 0);
str = fo.read(10)
print "Again read String is : ", str

# Reposition pointer at the beginning once again
position = fo.seek(2, 0);
str = fo.read(10)
print "Again read String is : ", str
# Close opend file
fo.close()
```

* mv, cp, rm, mkdir of files and directories

`os.rename(current_file_name, new_file_name)`

`os.remove(file_name)`

`os.mkdir("newdir")`

`os.chdir("newdir")`

`os.getcwd()`

`os.rmdir('dirname')`

https://www.techbeamers.com/python-file-handling-tutorial-beginners/
https://www.techbeamers.com/python-copy-file/

## Threading

### Start a thread

`thread.start_new_thread ( function, args[, kwargs] )`

### Threading module

These are similar to Java. Reference it when needed.

* threading.activeCount() âˆ’ Returns the number of thread objects that are active.
* threading.currentThread() âˆ’ Returns the number of thread objects in the caller's thread control.
* threading.enumerate() âˆ’ Returns a list of all thread objects that are currently active.

In addition to the methods, the threading module has the Thread class that implements threading. The methods provided by the Thread class are as follows:

`run()` âˆ’ The run() method is the entry point for a thread.

`start()` âˆ’ The start() method starts a thread by calling the run method.

`join([time])` âˆ’ The join() waits for threads to terminate.

`isAlive()` âˆ’ The isAlive() method checks whether a thread is still executing.

`getName()` âˆ’ The getName() method returns the name of a thread.

`setName()` âˆ’ The setName() method sets the name of a thread.

### Implement a thread class
Similar to Java

To implement a new thread using the threading module, you have to do the following âˆ’

* Define a new subclass of the Thread class.
* Override the __init__(self [,args]) method to add additional arguments.
* Then, override the run(self [,args]) method to implement what the thread should do when started.
* Once you have created the new Thread subclass, you can create an instance of it and then start a new thread by invoking the start(), which in turn calls run() method.

```
#!/usr/bin/python

import threading
import time

exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print "Starting " + self.name
      print_time(self.name, 5, self.counter)
      print "Exiting " + self.name

def print_time(threadName, counter, delay):
   while counter:
      if exitFlag:
         threadName.exit()
      time.sleep(delay)
      print "%s: %s" % (threadName, time.ctime(time.time()))
      counter -= 1

# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# Start new Threads
thread1.start()
thread2.start()

print "Exiting Main Thread"
```

### Sync threads

`Lock()`

`release()`

`acquire(blocking)`

```
#!/usr/bin/python

import threading
import time

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print "Starting " + self.name
      # Get lock to synchronize threads
      threadLock.acquire()
      print_time(self.name, self.counter, 3)
      # Free lock to release next thread
      threadLock.release()

def print_time(threadName, delay, counter):
   while counter:
      time.sleep(delay)
      print "%s: %s" % (threadName, time.ctime(time.time()))
      counter -= 1

threadLock = threading.Lock()
threads = []

# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# Start new Threads
thread1.start()
thread2.start()

# Add threads to thread list
threads.append(thread1)
threads.append(thread2)

# Wait for all threads to complete
for t in threads:
    t.join()
print "Exiting Main Thread"
```

### Multithreaded Priority Queue

The Queue module allows you to create a new queue object that can hold a specific number of items. There are following methods to control the Queue âˆ’

`get()` âˆ’ The get() removes and returns an item from the queue.

`put()` âˆ’ The put adds item to a queue.

`qsize()` âˆ’ The qsize() returns the number of items that are currently in the queue.

`empty()` âˆ’ The empty( ) returns True if queue is empty; otherwise, False.

`full()` âˆ’ the full() returns True if queue is full; otherwise, False.

```
#!/usr/bin/python

import Queue
import threading
import time

exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, threadID, name, q):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.q = q
   def run(self):
      print "Starting " + self.name
      process_data(self.name, self.q)
      print "Exiting " + self.name

def process_data(threadName, q):
   while not exitFlag:
      queueLock.acquire()
         if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print "%s processing %s" % (threadName, data)
         else:
            queueLock.release()
         time.sleep(1)

threadList = ["Thread-1", "Thread-2", "Thread-3"]
nameList = ["One", "Two", "Three", "Four", "Five"]
queueLock = threading.Lock()
workQueue = Queue.Queue(10)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
   thread = myThread(threadID, tName, workQueue)
   thread.start()
   threads.append(thread)
   threadID += 1

# Fill the queue
queueLock.acquire()
for word in nameList:
   workQueue.put(word)
queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
   pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()
print "Exiting Main Thread"

```

## Regex

### regex match
`re.match(pattern, string, flags=0)`

### regex search
`asdf`

### regex place
`re.sub(pattern, repl, string, max=0)`

### optional flags

`flags`: You can specify different flags using bitwise OR (|). These are modifiers, which are listed in the table below.
The modifiers are specified as an optional flag. You can provide multiple modifiers using exclusive OR (|), as shown previously and may be represented by one of these âˆ’

`re.I`: Performs case-insensitive matching.

`re.L`: Interprets words according to the current locale. This interpretation affects the alphabetic group (\w and \W), as well as word boundary behavior(\b and \B).

`re.M`: Makes $ match the end of a line (not just the end of the string) and makes ^ match the start of any line (not just the start of the string).

`re.S`: Makes a period (dot) match any character, including a newline.

`re.U`: Interprets letters according to the Unicode character set. This flag affects the behavior of \w, \W, \b, \B.

`re.X`: Permits "cuter" regular expression syntax. It ignores whitespace (except inside a set [] or when escaped by a backslash) and treats unescaped # as a comment marker.

### Control characters

* Basics

Control characters, ( + ? . * ^ $ ( ) [ ] { } | \ ), to escape control characters, put a backslash \ in the front.

`^:` begining of the line

`$:` end of the line

`.:` any single char except newline, using m option to make it match new line as well.

`[]`: match single characters in the branch, which defines a ring.

`[^]`: match single characters not in the brackets

`re*`: 0 or more occurence of expr `re`

`re+`: 1 or more occurence of expr `re`

`re?`: 0 or 1 occurence of expr `re`

`re{n}`: match exact n repeating of re

`re{n,}`: match n and more repeating

`re{n,m}`: match between n and m times

`a|b`: matches either a or b

`(re)`: groups regular expressions and remembers matched text.

`\w`: word

`\W`: nonword

`\s`: whitespace, such as \t\n\r\f

`\S`: not whitespace

`\d`: digit

`\D`: non digit

`\A`: Match begining of string

`\Z`: Match end of string. If new line exists, it matches just before newline.

`\z`: Match end of string.

`\G`: match point where last match finished

`\B`: non word boundary

`\n,\t`: match new line, carriage returns, tabs...etc

`\1...\9`: matches nth grouped subexpression.

`\10`: Matches nth grouped subexpression if it matched already. Otherwise refers to the octal representation of a character code.

* Advanced

(?imx): Temporarily toggles on i, m, or x options within a regular expression. If in parentheses, only that area is affected.

(?-imx): Temporarily toggles off i, m, or x options within a regular expression. If in parentheses, only that area is affected.

(?: re): Groups regular expressions without remembering matched text.

(?imx: re): Temporarily toggles on i, m, or x options within parentheses.

(?-imx: re): Temporarily toggles off i, m, or x options within parentheses.

(?#...): Comment.

(?= re): Specifies position using a pattern. Doesn't have a range.

(?! re): Specifies position using pattern negation. Doesn't have a range.

(?> re): Matches independent pattern without backtracking.

Example:

```
#!/usr/bin/python
import re

## Greedy repetion example

line = "<python>perl>"

matchObj = re.match( r'<.*>', line, re.M|re.I)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
  
else:
   print "No match!!"

## Non greedy repetion example, get the first one, and stop

matchObj = re.match( r'<.*?>', line, re.M|re.I)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
  
else:
   print "No match!!"
   
line = "<<python>perl>"

matchObj = re.match( r'<.*?>', line, re.M|re.I)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```

* Grouping

```
(\D\d)+
([Pp]ython(, )?)+
```

* Backreferences

```
([Pp])ython&\1ails

Match python&pails or Python&Pails
	
(['"])[^\1]*\1

Single or double-quoted string. \1 matches whatever the 1st group matched. \2 matches whatever the 2nd group matched, etc.
```

* specify options for subexpr

?i means the same as the re.I option is for global, but ?i can be applied to subexpr or string before it.

```
## apply options works the same as the $
print("testing ?i");
line = "ruby"
matchObj = re.match(r'R(?i)uby', line, re.M)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```

?! to negate some symbol after it.

```
print("testing ?!");
line = "Python"
matchObj = re.match(r'Python(?!!)', line, re.M)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```

?= means followed by something,

```
print("testing ?=");
line = "Python!"
matchObj = re.match(r'Python(?=!)', line, re.M)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```

## XML

Python had two libs that support XML,

* SAX, simple API for XML, and this is not reading entire file into memory. SAX is slower than DOM API with large files. SAX is read only.
* DOM API, tree-based structure and reads entire file into memory. DOM can kill your resources if used on a lot of small files. DOM can write to it.
  For large project, you might need to use both of them.

`DOM example`: SAX example is skipped, please check example code with this tutorial.

```
#!/usr/bin/python

from xml.dom.minidom import parse
import xml.dom.minidom

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse("movies.xml")
collection = DOMTree.documentElement
if collection.hasAttribute("shelf"):
   print "Root element : %s" % collection.getAttribute("shelf")

# Get all the movies in the collection
movies = collection.getElementsByTagName("movie")

# Print detail of each movie.
for movie in movies:
   print "*****Movie*****"
   if movie.hasAttribute("title"):
      print "Title: %s" % movie.getAttribute("title")

   type = movie.getElementsByTagName('type')[0]
   print "Type: %s" % type.childNodes[0].data
   format = movie.getElementsByTagName('format')[0]
   print "Format: %s" % format.childNodes[0].data
   rating = movie.getElementsByTagName('rating')[0]
   print "Rating: %s" % rating.childNodes[0].data
   description = movie.getElementsByTagName('description')[0]
   print "Description: %s" % description.childNodes[0].data
```

## Database

Python has many modules|lib of database drivers, and `MySQLdb` is one of them. Install it with Pip.
Very similar to other languages such as PHP and Java.

### Core operations
`MySQLdb.connect(host,usr,pwd,dbname)`,`db.cursor()`,`cursor.execute(sql)`,`db.commit()`, `db.rollback()`, `db.close()`

```
#!/usr/bin/python

import MySQLdb

# Open database connection
db = MySQLdb.connect("localhost","testuser","test123","TESTDB" )

# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT VERSION()")

# Fetch a single row using fetchone() method.
data = cursor.fetchone()
print "Database version : %s " % data


### Create table

# Drop table if it already exist using execute() method.
cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")

# Create table as per requirement
sql = """CREATE TABLE EMPLOYEE (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,  
         SEX CHAR(1),
         INCOME FLOAT )"""

cursor.execute(sql)


# insert

# Prepare SQL query to INSERT a record into the database.
sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
         LAST_NAME, AGE, SEX, INCOME)
         VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Commit your changes in the database
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()
   
# Insert from variables
# Prepare SQL query to INSERT a record into the database.
sql = "INSERT INTO EMPLOYEE(FIRST_NAME, \
       LAST_NAME, AGE, SEX, INCOME) \
       VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
       ('Mac', 'Mohan', 20, 'M', 2000)
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Commit your changes in the database
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()

# SQL read

sql = "SELECT * FROM EMPLOYEE \
       WHERE INCOME > '%d'" % (1000)
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Fetch all the rows in a list of lists.
   results = cursor.fetchall()
   for row in results:
      fname = row[0]
      lname = row[1]
      age = row[2]
      sex = row[3]
      income = row[4]
      # Now print fetched result
      print "fname=%s,lname=%s,age=%d,sex=%s,income=%d" % \
             (fname, lname, age, sex, income )
except:
   print "Error: unable to fecth data"
   
# SQL update

# Prepare SQL query to UPDATE required records
sql = "UPDATE EMPLOYEE SET AGE = AGE + 1
                          WHERE SEX = '%c'" % ('M')
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Commit your changes in the database
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()

# SQL delete

# Prepare SQL query to DELETE required records
sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Commit your changes in the database
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()

# disconnect from server
db.close()
```

## Networking

Socket, similar to Java.

`s = socket.socket (socket_family, socket_type, protocol=0)`

`socket_family`: This is either AF_UNIX or AF_INET, as explained earlier.

`socket_type`: This is either SOCK_STREAM or SOCK_DGRAM.

`protocol`: This is usually left out, defaulting to 0.

### Server socket methods

* `s.bind()`: This method binds address (hostname, port number pair) to socket.
* `s.listen()`: This method sets up and start TCP listener.
* `s.accept()`: This passively accept TCP client connection, waiting until connection arrives (blocking).

### Client Socket methods

* s.connect(): This method actively initiates TCP server connection.

### General APIs

* `s.recv()`: This method receives TCP message
* `s.send()`: This method transmits TCP message
* `s.recvfrom()`: This method receives UDP message
* `s.sendto()`: This method transmits UDP message
* `s.close()`: This method closes socket
* `socket.gethostname()`: Returns the hostname.

## Server client example

Server

```
#!/usr/bin/python           # This is server.py file

import socket               # Import socket module

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345                # Reserve a port for your service.
s.bind(('localhost', port))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
while True:
   c, addr = s.accept()     # Establish connection with client.
   print 'Got connection from', addr
   c.send('Thank you for connecting')
   c.close()                # Close the connection
```

client

```
#!/usr/bin/python           # This is client.py file

import socket               # Import socket module

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345                # Reserve a port for your service.

s.connect(('localhost', port))
print s.recv(1024)
s.close()                   # Close the socket when done
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

https://www.tutorialspoint.com/python
https://www.techbeamers.com/python-generator/
