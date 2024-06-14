# Python Ref - Name, Scope, Function, Class and Module

## Name and Scope

Scope are similar to Java and C, however, there is a keyword `global` where inside of a function, you can declare a variable is global,
or if you want to change the global var's value, you can claim it is global.

```python
def myfunc():
  global x
  x = 300

myfunc()

print(x)

```

```python
x = 300

def myfunc():
  global x
  x = 200

myfunc()

print(x)
```

https://www.techbeamers.com/python-namespace-scope/

## Function

```python
def my_function():
  print("Hello from a function")
```

revoke function

`my_function()`

## Class

```python
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
    
  def myfunc(self):
    print("Hello my name is " + self.name)
```

init class and use class,

```python
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

```python
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

- https://www.techbeamers.com/python-function/
- https://www.techbeamers.com/python-class/
- https://www.techbeamers.com/python-inheritance/

## Module

Module is the concept of library, it deals with `from`, `import`. Normally a library has extension of `.py`.
`from xxx import yyy` is similar to Java's static import. However, you can not write `import yyy from xxx`.

mymodule.py

```python
def greeting(name):
  print("Hello, " + name)
```

hello-world.py

```python
import mymodule
def my_function(name):
    mymodule.greeting(name)

my_function("xiaofeng")
```

```python
from mymodule import greeting
def my_function(name):
    greeting(name)

my_function("xiaofeng")
```

https://www.techbeamers.com/python-module/