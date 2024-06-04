# Java Stream Tutorial

This tutorial provides an overview of Java Streams, introduced in JDK 8, along with various stream operations and examples.

## Introduction

Streams provide a modern and more concise way to work with collections in Java. They allow for declarative operations on data, making code more expressive and readable.

## Code Overview

The `StreamTutorial` class contains examples demonstrating the usage of streams and various stream operations:

### Stream Creation

- **Creating Streams from Arrays:**
  - Streams can be created from arrays using `Stream.of()` or by converting arrays to lists and then obtaining a stream.
- **Stream Builder:**
  - Stream builder allows for the dynamic construction of streams by adding elements individually.
- **Converting Streams to Arrays:**
  - Streams can be converted to arrays using the `toArray` method.

### Stream Operations

- **forEach:**
  - `forEach` is used to iterate over elements of a stream and perform an action.
- **peek:**
  - `peek` is similar to `forEach`, but it allows chaining multiple operations together.
- **map:**
  - `map` transforms each element of a stream using a provided function.
- **flatMap:**
  - `flatMap` is used to flatten nested collections within a stream.
- **filter:**
  - `filter` selects elements from a stream based on a specified condition.
- **findFirst and orElse:**
  - `findFirst` returns the first element of a stream, while `orElse` provides a default value if the stream is empty.
- **orElseThrow:**
  - `orElseThrow` throws an exception if the stream is empty.
- **count:**
  - `count` terminal operation returns the number of elements in the stream.

### Method References and Lambdas

- **Method References:**
  - Method references (`System.out::println`) provide a shorthand notation for invoking methods.
- **Lambda Expressions:**
  - Lambda expressions (`e -> e.salaryIncrement(10.0)`) allow for concise function definitions inline.

### Stream Pipelines

- **Pipeline Structure:**
  - A stream pipeline consists of a stream source, followed by zero or more intermediate operations, and a terminal operation.
- **Intermediate and Terminal Operations:**
  - Intermediate operations (e.g., `map`, `filter`) produce a new stream, while terminal operations (e.g., `forEach`, `count`) produce a result or side-effect.

### Short-circuiting Operations

- **Short-circuiting Behavior:**
  - Short-circuiting operations like `findFirst` allow computations on infinite streams to complete in finite time by processing only the necessary elements.

## Conclusion

Java Streams provide a powerful mechanism for processing collections in a functional and expressive manner. By understanding and leveraging streams and their operations, developers can write cleaner, more efficient, and more maintainable code.

