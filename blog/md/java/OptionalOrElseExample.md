# Java Optional orElse and orElseGet Example

This example demonstrates the usage of the `orElse` and `orElseGet` methods of the `Optional` class in Java. These methods provide a way to specify a default value to return if the `Optional` is empty.

## Java Code

```java
package com.webbertech.java;

import java.util.Optional;

public class OptionalOrElseExample {

    public OptionalOrElseExample() {
        // TODO Auto-generated constructor stub
    }

    public static void main(String[] args) {

        Optional<String> gender = Optional.of("MALE");
        Optional<String> emptyGender = Optional.empty();

        System.out.println(gender.orElse("<N/A>")); // MALE
        System.out.println(emptyGender.orElse("<N/A>")); // <N/A>

        System.out.println(gender.orElseGet(() -> "<N/A>")); // MALE
        System.out.println(emptyGender.orElseGet(() -> "<N/A>")); // <N/A>

    }

}
