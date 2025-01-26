# Python implementation of Breadth-First Search (BFS) for graph traversal using an adjacency list representation


```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        current_node = queue.popleft()
        result.append(current_node)
        
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

# Example usage:
if __name__ == "__main__":
    # Sample graph (adjacency list)
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

    print("BFS traversal starting from 'A':")
    print(bfs(graph, 'A'))  # Output: ['A', 'B', 'C', 'D', 'E', 'F']
```


**Key components explained:**

1. Queue: Uses deque for efficient FIFO operations (O(1) for append/popleft)
2. Visited Set: Tracks visited nodes to prevent revisiting
3. Traversal Order: Stores nodes in the order they are visited

**How it works:**

1. Starts from the given node, marks it as visited
2. Processes nodes level by level (breadth-first)
3. Adds unvisited neighbors to the queue for subsequent processing

**Notes:**

* Handles both directed and undirected graphs
* Returns nodes in BFS order
* Time complexity: O(V + E) (vertices + edges)
* Space complexity: O(V)