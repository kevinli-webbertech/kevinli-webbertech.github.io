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

- https://www.tutorialspoint.com/python
- https://www.techbeamers.com/python-generator/
