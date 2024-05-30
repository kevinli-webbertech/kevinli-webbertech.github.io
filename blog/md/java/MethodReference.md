# Java Method Reference Example

This example demonstrates the use of method references in Java. Method references provide a shorthand syntax for calling methods directly and can make the code more concise and readable.

## Java Code

```java
package com.webbertech.java;

import java.util.Arrays;
import java.util.List;

public class MethodReference {
    public static void main(String args[]) {
        // This is the usage of a convenient util
        List<String> names = Arrays.asList("Mahesh", "Suresh", "Ramesh", "Naresh", "Kalpesh");
        
        // The following line is the usage of the method reference
        names.forEach(System.out::println);
    }
}
