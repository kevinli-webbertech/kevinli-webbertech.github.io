# Python Ref - Lambda
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
