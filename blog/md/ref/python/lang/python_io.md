# Python Ref - IO

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

```python
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

- <https://www.techbeamers.com/python-file-handling-tutorial-beginners/>
- <https://www.techbeamers.com/python-copy-file/>
