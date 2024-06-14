
# Python Ref - Control and Loop

**if**

<b>if and io operation </b>

```
days = int(input("How many days in a leap year? "))
if days == 366:
    print("You have cleared the test.")
print("Congrats!")
```

**if/else**

```python
answer = input("Is Python an interpreted language? Yes or No >> ").lower()

if answer == "yes" :
    print("You have cleared the test.")
else :
    print("You have failed the test.")

print("Thanks!")
```

**if/else on one line**

If Else in one line - Syntax

`value_on_true if condition else value_on_false`

Example,

```python
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

**if/elif/else**

```python
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

**nested if example**

```python
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

**Using Not Operator With Python If Else**

```python
a = 10
b = 20
if not a > b :
    print("The number %d is less than %d" %(a, b))/
```

## Switch Case

https://www.techbeamers.com/python-switch-case-statement/

## Loops


**for in**
```python
for iter in sequence:
    statements(iter)
```

Example,

```python
vowels="AEIOU"
for iter in vowels:
    print("char:", iter)
```     

Another example,

```python
int_list = [1, 2, 3, 4, 5, 6]
sum = 0
for iter in int_list:
    sum += iter
print("Sum =", sum)
print("Avg =", sum/len(int_list))
```

**Range() function**
A similar function is also seen in Mathematica.

Example,

```python
for iter in range(0, 3):
    print("iter: %d" % (iter))
```

Another example

```python
>>> books = ['C', 'C++', 'Java', 'Python']
>>> for index in range(len(books)):
...     print('Book (%d):' , index, books[index])
```

*Else Clause*
Interestingly, Python allows using an optional else statement along with the â€œforâ€� loop.

The code under the else clause executes after the completion of the â€œforâ€� loop. However, if the loop stops due to a â€œbreakâ€� call, then itâ€™ll skip the â€œelseâ€� clause.

**For-Else Syntax**

```python

for item in seq:
    statement 1
    statement 2
    if <cond>:
        break
else:
    statements
    
```

## While loop

https://www.techbeamers.com/python-while-loop/

## Iterators

* In Python, an iterator is an object which implements the iterator protocol, which consist of the methods __iter__() and __next__().
* Lists, tuples, dictionaries, and sets are all iterable objects. They are iterable containers which you can get an iterator from.
* All these objects have a iter() method which is used to get an iterator.

```python
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))
```

Same thing for string mystr = "banana", all the above can be done by for loop,

```python
mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)
```

and

```python
mystr = "banana"

for x in mystr:
  print(x)
```
