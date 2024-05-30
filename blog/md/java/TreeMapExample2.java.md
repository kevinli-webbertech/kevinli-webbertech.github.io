# TreeMap Example with Custom Objects

This example demonstrates the usage of `TreeMap` with custom objects (`Dog`) as keys.

## Introduction

`TreeMap` is a sorted map implementation in Java, where keys are ordered based on their natural ordering or a custom comparator. This example showcases how to use `TreeMap` with custom objects and provides insights into how the sorting mechanism works.

## Code Overview

The `TreeMapExample2` class contains the main method to demonstrate the usage of `TreeMap` with custom objects:

```java
package com.webbertech.java;

import java.util.Map.Entry;
import java.util.TreeMap;

public class TreeMapExample2 {

    public static void main(String[] args) {
        Dog d1 = new Dog("red", 30);
        Dog d2 = new Dog("black", 20);
        Dog d3 = new Dog("white", 10);
        Dog d4 = new Dog("white", 10);
        
        TreeMap<Dog, Integer> treeMap = new TreeMap<>();
        treeMap.put(d1, 10);
        treeMap.put(d2, 15);
        treeMap.put(d3, 5);
        treeMap.put(d4, 20);
        
        for (Entry<Dog, Integer> entry : treeMap.entrySet()) {
            System.out.println(entry.getKey() + " - " + entry.getValue());
        }
    }
}

class Dog implements Comparable<Dog> {
    String color;
    int size;

    Dog(String c, int s) {
        color = c;
        size = s;
    }

    public String toString() {
        return color + " dog";
    }

    @Override
    public int compareTo(Dog o) {
        return o.size - this.size;
    }
}
