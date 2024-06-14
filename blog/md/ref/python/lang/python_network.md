# Python Ref - Networking

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

**Server**

```python
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

**client**

```python
#!/usr/bin/python           # This is client.py file

import socket               # Import socket module

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345                # Reserve a port for your service.

s.connect(('localhost', port))
print s.recv(1024)
s.close()                   # Close the socket when done
```