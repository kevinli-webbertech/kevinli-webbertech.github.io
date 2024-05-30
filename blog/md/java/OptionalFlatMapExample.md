# Java Optional FlatMap Example

This example demonstrates the usage of `Optional` in Java, particularly focusing on the `map` and `flatMap` methods. These methods allow transforming the value contained within an `Optional` in different ways, depending on whether the transformation produces another `Optional`.

## Java Code

```java
package com.webbertech.java;

import java.util.Optional;

public class OptionalFlatMapExample {

    public OptionalFlatMapExample() {
        // TODO Auto-generated constructor stub
    }

    public static void main(String[] args) {

        Optional<String> nonEmptyGender = Optional.of("male");
        Optional<String> emptyGender = Optional.empty();

        System.out.println("Non-Empty Optional:: " + nonEmptyGender.map(String::toUpperCase));
        System.out.println("Empty Optional    :: " + emptyGender.map(String::toUpperCase));

        Optional<Optional<String>> nonEmptyOptionalGender = Optional.of(Optional.of("male"));
        System.out.println("Optional value   :: " + nonEmptyOptionalGender);
        System.out.println("Optional.map     :: " + nonEmptyOptionalGender.map(gender -> gender.map(String::toUpperCase)));
        System.out.println("Optional.flatMap :: " + nonEmptyOptionalGender.flatMap(gender -> gender.map(String::toUpperCase)));

    }

}
