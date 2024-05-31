# Java Optional ifPresent

This example demonstrates the usage of the `Optional` class in Java, specifically focusing on the `ifPresent` method. The `ifPresent` method allows executing a block of code only if the `Optional` contains a non-null value.

```java
package com.webbertech.java;

import java.util.Optional;

public class OptionalIfPresentExample {

    public OptionalIfPresentExample() {
        // TODO Auto-generated constructor stub
    }

    public static void main(String[] args) {

        Optional<String> gender = Optional.of("MALE");
        Optional<String> emptyGender = Optional.empty();

        if (gender.isPresent()) {
            System.out.println("Value available.");
        } else {
            System.out.println("Value not available.");
        }

        gender.ifPresent(g -> System.out.println("In gender Option, value available."));

        // condition failed, no output print
        emptyGender.ifPresent(g -> System.out.println("In emptyGender Option, value available."));

    }

}
