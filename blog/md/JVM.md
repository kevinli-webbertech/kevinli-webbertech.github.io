# JVM (Java Virtual Machine) Architecture

## What is JVM?
The JVM (Java Virtual Machine) is an abstract machine that provides a runtime environment to execute Java bytecode.

## Key Points
- The JVM is a specification with multiple implementations (JRE).
- It loads, verifies, and executes Java bytecode.
- The JVM consists of various components like class loaders, memory areas, and execution engines.
- Understanding JVM internals helps in debugging and optimizing Java applications.

## Java Code Execution Process
#### 1. **Class Loader**: Loads Java bytecode into runtime memory areas.
#### 2. **Runtime Data Areas**: Memory areas where the bytecode is loaded.
#### 3. **Execution Engine**: Executes the Java bytecode.

### Class Load Stages
#### 1. **Loading**: Obtains the class and loads it into memory.

***Example of JVM ClassLoader:***

When you run a Java program, the JVM loads the required classes using its class loaders. Here's an example that shows the class loader hierarchy:

```java
public class ClassLoaderExample {
    public static void main(String[] args) {
        Class<?> cls = ClassLoaderExample.class;
        System.out.println("ClassLoader of this class: " + cls.getClassLoader());

        Class<?> stringClass = String.class;
        System.out.println("ClassLoader of String class: " + stringClass.getClassLoader());
    }
}
```

**Output:**

```
ClassLoader of this class: sun.misc.Launcher$AppClassLoader@4e0e2f2a
ClassLoader of String class: null
```

***Note:*** This output indicates that the `ClassLoaderExample` class is loaded by the application class loader, while the `String` class, which is part of the Java core libraries, is loaded by the bootstrap class loader.

#### 2. **Verifying**: Ensures the class file is correct and safe.
#### 3. **Preparing**: Allocates memory for class variables and sets default values.
#### 4. **Resolving**: Converts symbolic references to direct references.
#### 5. **Initializing**: Initializes static variables and runs static blocks.

### Runtime Data Areas
Memory areas used by JVM during execution:

#### 1. **Method Area**: Stores class-level data like fields and methods.

### Class (Method) Area Example

The Class (Method) Area stores per-class structures like the runtime constant pool, field, and method data. This is more about the JVM internals, so here's a simple example to understand that static methods and fields are stored in the Method Area.

```java
public class MethodAreaExample {
    public static void main(String[] args) {
        System.out.println("Static method called");
        System.out.println("Static field: " + MyClass.staticField);
    }
}
```

#### 2. **Heap**: Stores all objects and instance variables.

***Heap Example:***

```java
public class HeapExample {
    public static void main(String[] args) {
        MyClass obj1 = new MyClass();
        MyClass obj2 = new MyClass();
        System.out.println("Objects created on the heap: " + obj1 + ", " + obj2);
    }
}

class MyClass {
    int instanceVariable;
}
```

#### 3. **Stack**: Stores method call frames, local variables, and partial results.

***Stack Example 1:***

The Stack stores frames, holding local variables and partial results. Each thread has its own JVM stack:

```java
public class StackExample {
    public static void main(String[] args) {
        method1();
    }

    public static void method1() {
        int x = 10;
        method2(x);
    }

    public static void method2(int y) {
        int z = y * 2;
        System.out.println("Result: " + z);
    }
}
```

***Stack Frame Configuration Example 2:***

```java
public class StackExample {
    public static void main(String[] args) {
        double score = calculateScore();
        System.out.println(normalizeScore(score));
    }

    public static double calculateScore() {
        // some calculation
        return 100.0;
    }

    public static double normalizeScore(double score) {
        double minScore = 0.0;
        double maxScore = 200.0;
        return (score - minScore) / (maxScore - minScore);
    }
}
```


#### 4. **PC (Program Counter) Register**: Holds the address of the current instruction.
#### 5. **Native Method Stack**: Supports native methods written in other languages.

***Native Method Stack Example:***

The Native Method Stack contains all the native methods used in the application:

```java
public class NativeMethodExample {
    static {
        System.loadLibrary("nativeLib");
    }

    private native void nativeMethod();

    public static void main(String[] args) {
        new NativeMethodExample().nativeMethod();
    }
}
```

**Note:** The above code requires a corresponding native library (`nativeLib`) and native method implementation, which would be written in C/C++.


#### 7. **Execution Engine**
    - Executes the bytecode.
    1.  **Interpreter**: Reads and executes bytecode instructions.

***Execution Engine Example:***

The Execution Engine includes an interpreter and JIT compiler. Here’s an example illustrating bytecode execution and JIT compilation:

```java
public class ExecutionEngineExample {
    public static void main(String[] args) {
        int sum = 0;
        for (int i = 0; i < 1000; i++) {
            sum += i;
        }
        System.out.println("Sum: " + sum);
    }
}
```

#### 2. **Just-In-Time (JIT) Compiler**: Compiles bytecode into native code for performance.

***Example: JIT Compilation:***

```java
public class JITExample {
    public static void main(String[] args) {
        int sum = 10;
        for (int i = 0; i <= 10; i++) {
            sum += i;
        }
        System.out.println(sum);
    }
}
```
The JIT compiler optimizes this code by compiling the loop to native code, reducing memory access overhead.

### Garbage Collector
Automatically reclaims memory by removing unreferenced objects.

### Types of Garbage Collectors
#### 1. **Serial GC**: Single-threaded, for small applications.
#### 2. **Parallel GC**: Multi-threaded, default in JVM.
#### 3. **G1 GC (Garbage First)**: For large heap sizes, prioritizes garbage collection in regions with the most garbage.



#### 8. **Java Native Interface (JNI)**
    - Allows Java code to interact with code written in other languages like C and C++.

### Java Native Interface (JNI) Example

```java
public class JniExample {
    static {
        System.loadLibrary("nativeLib");
    }

    private native void nativeMethod();

    public static void main(String[] args) {
        new JniExample().nativeMethod();
    }
}
```

**Note:** This requires a native implementation of `nativeMethod` in a library named `nativeLib`.

### Common JVM Errors
#### 1. **ClassNotFoundException**: Class loader can’t find the class.
#### 2. **NoClassDefFoundError**: Class file found during compile time but not at runtime.
#### 3. **OutOfMemoryError**: JVM runs out of memory.
#### 4. **StackOverflowError**: JVM stack exceeds its limit.

### Conclusion
Understanding the JVM's structure helps in writing efficient Java code and troubleshooting runtime issues. While many Java developers may not delve deep into the JVM internals, having this knowledge can be incredibly beneficial for optimizing and debugging Java applications.

