# TreeMap vs. HashMap Example

This example demonstrates the differences between `HashMap` and `TreeMap` in Java, focusing on their ordering properties.

## Introduction

`HashMap` and `TreeMap` are both implementations of the `Map` interface in Java. However, they have different characteristics in terms of ordering and performance.

### HashMap

- `HashMap` is an unordered collection that stores key-value pairs.
- It uses hash table data structure to store elements.
- The elements are not sorted in any particular order.
- It offers constant-time performance for basic operations (add, remove, get), assuming the hash function disperses elements properly.

### TreeMap

- `TreeMap` is a Red-Black tree-based implementation of the `NavigableMap` interface.
- It stores key-value pairs in sorted order based on the natural ordering of its keys.
- The elements are sorted in ascending order of their keys.
- It provides guaranteed log(n) time cost for the containsKey, get, put, and remove operations.

## Code Overview

The `TreeMapExample` class demonstrates the usage of `HashMap` and `TreeMap`, showcasing their differences in ordering:

```java
package com.webbertech.java;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.TreeMap;

public class TreeMapExample {
    public static void main(String args[]) {
        // Creating HashMap
        HashMap<Integer, String> hm = new HashMap<Integer, String>();
        hm.put(1, "Data1");
        hm.put(23, "Data2");
        hm.put(70, "Data3");
        hm.put(4, "Data4");
        hm.put(2, "Data5");
        
        // Printing HashMap
        System.out.println("HashMap:");
        for (Entry<Integer, String> m : hm.entrySet()) {
            System.out.println(m.getKey() + " " + m.getValue());
        }
        
        // Creating TreeMap from HashMap
        TreeMap<Integer, String> tmap = new TreeMap<>(hm);
        
        // Printing TreeMap
        System.out.println("TreeMap:");
        for (Entry<Integer, String> m : tmap.entrySet()) {
            System.out.println(m.getKey() + " " + m.getValue());
        }
    }   
}
