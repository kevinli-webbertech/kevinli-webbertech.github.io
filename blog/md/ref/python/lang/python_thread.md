# Python Ref - Thread

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

```python
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

```python
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

`get()` The get() removes and returns an item from the queue.

`put()`  The put adds item to a queue.

`qsize()` The qsize() returns the number of items that are currently in the queue.

`empty()` The empty( ) returns True if queue is empty; otherwise, False.

`full()` the full() returns True if queue is full; otherwise, False.

```python
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
