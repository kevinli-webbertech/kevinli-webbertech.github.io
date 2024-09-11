# Cassandra Programming

## With Python

## Install driver

`pip3 install cassandra-driver`

Ref:

- https://www.tutorialspoint.com/python_data_persistence/python_data_persistence_cassandra_driver.htm

## Code

```python
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.auth import PlainTextAuthProvider

auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')

clstr=Cluster(['0.0.0.0'] ,
                #load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='US-WEST'),
                port=9042,
                auth_provider=auth_provider
                )
session=clstr.connect()
session.execute("create keyspace if not exists mykeyspace with replication={'class':'SimpleStrategy','replication_factor':1};")
session=clstr.connect('mykeyspace')
qry= '''
 create table if not exists students (
  studentID int,
   name text,
   age int,
   marks int,
   primary key(studentID)
);'''
session.execute(qry)
session.execute("insert into students (studentID, name, age, marks) values (1, 'Juhi',20, 200);")
rows=session.execute("select * from students;")
for row in rows:
    print ('StudentID: {} Name:{} Age:{}'.format(row[0],row[1], row[2]))
    #print(row)
```