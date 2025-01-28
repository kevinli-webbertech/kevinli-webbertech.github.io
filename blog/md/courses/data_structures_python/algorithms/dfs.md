# Python implementation of Depth-First Search (DFS) with both iterative and recursive approaches:

```python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    visited.add(start)
    result = []
    
    while stack:
        current_node = stack.pop()
        result.append(current_node)
        
        # Process neighbors in reverse order to match recursive DFS sequence
        for neighbor in reversed(graph[current_node]):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    
    return result

def dfs_recursive(graph, start):
    visited = set()
    result = []
    
    def dfs_helper(node):
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                dfs_helper(neighbor)
    
    dfs_helper(start)
    return result

# Example usage:
if __name__ == "__main__":
    # Same graph as BFS example
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

    print("DFS iterative starting from 'A':")
    print(dfs_iterative(graph, 'A'))  # Output: ['A', 'B', 'D', 'E', 'F', 'C']
    
    print("\nDFS recursive starting from 'A':")
    print(dfs_recursive(graph, 'A'))  # Output: ['A', 'B', 'D', 'E', 'F', 'C']
```

**Key components explained:**

**Iterative DFS:**

1. Uses a stack (LIFO) for frontier management
2. Processes nodes by depth-first order
3. Marks nodes visited when first encountered
4. Neighbors processed in reverse order to match recursive sequence

**Recursive DFS:**

1. Uses implicit call stack
2. Naturally implements depth-first behavior
3. Visits nodes in first-neighbor-first order

**Key differences from BFS:**

* Uses stack instead of queue (LIFO vs FIFO)
* Explores depth-first rather than breadth-first
* Order of neighbor processing affects traversal sequence

**Complexity:**

* Time: O(V + E) (vertices + edges)
* Space: O(V) for visited set and stack/call-stack

**Important Notes:**

1. The iterative version processes neighbors in reverse order (reversed()) to match the recursive implementation's sequence
2. For large graphs, prefer iterative approach to avoid recursion depth limits
3. Both implementations handle cycles through the visited set
4. The recursive version maintains state through closure variables
5. Output order may vary with different neighbor processing orders