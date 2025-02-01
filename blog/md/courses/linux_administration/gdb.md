# GDB

## Overview

* Create a binary with debugging info using `gcc`
* Debug binary using `gdb`
* `nm` util's usage
* `ldd` 's usage

## **Using `gcc` for Debugging in Linux**

When developing in C, debugging is crucial for identifying and fixing errors. The GNU Debugger (`gdb`) is commonly used alongside `gcc` to debug programs.

---

### **1. Compile with Debugging Information**
To enable debugging, compile with the `-g` flag:
```bash
gcc -g program.c -o program
```
- `-g` → Includes debugging symbols.

---

### **2. Run the Program in `gdb`**
Start debugging:
```bash
gdb ./program
```

---

### **3. Setting Breakpoints**
Inside `gdb`, you can set breakpoints:
```gdb
break main
```
or on a specific line:
```gdb
break 10
```

---

### **4. Running the Program**
Start execution:
```gdb
run
```

---

### **5. Stepping Through Code**
- Step into a function:
  ```gdb
  step
  ```
- Step over (execute current line and move to the next):
  ```gdb
  next
  ```

---

### **6. Printing Variables**
- Print a variable’s value:
  ```gdb
  print x
  ```
- Print all local variables:
  ```gdb
  info locals
  ```

---

### **7. Backtrace (Check Call Stack)**
If the program crashes, use:
```gdb
backtrace
```
to see the function call history.

---

### **8. Continue Execution**
Resume execution after a breakpoint:
```gdb
continue
```

---

### **9. Quit `gdb`**
Exit debugging:
```gdb
quit
```

## **`nm`

### **`nm` (List Symbols in a Binary or Library)**

The `nm` command displays the symbol table of an object file, executable, or library.

#### **Usage Examples**

- List symbols in an object file:
  ```bash
  nm math_operations.o
  ```
- List symbols in a shared library:
  ```bash
  nm -D libmath.so
  ```
- Filter only defined functions:
  ```bash
  nm -g --defined-only program
  ```

#### **Symbol Types**

- `T` → Text (code) section (global function)
- `U` → Undefined symbol (external function)
- `D` → Data section (global variable)
- `B` → BSS section (uninitialized global variable)

---

## **`ldd` (List Shared Library Dependencies)**

The `ldd` command shows the shared libraries required by a binary.

### **Usage Examples**

- Check dependencies of an executable:
  ```bash
  ldd /bin/ls
  ```
- Check dependencies of a compiled program:
  ```bash
  ldd ./program
  ```
- Check dependencies of a shared library:
  ```bash
  ldd libmath.so
  ```

