# Python Ref - Exception


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

## try,except,else paradigm

Similar to Java, but a little different, it is `else` is not `finally`.

```python
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

```python
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

```python
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

```python
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

```python
try:
   You do your operations here;
   ......................
   Due to any exception, this may be skipped.
finally:
   This would always be executed.
   ......................
```

Example

```python
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

```python
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

```python
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

```python
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

- https://www.techbeamers.com/python-copy-file/
- https://www.techbeamers.com/use-try-except-python/
