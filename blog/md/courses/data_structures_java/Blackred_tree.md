A **Red-Black Tree** (RB Tree) is a type of self-balancing binary search tree. It provides efficient insertion, deletion, and search operations, with a time complexity of O(log n). A Red-Black Tree ensures that the tree remains balanced by enforcing a set of properties, allowing the tree to remain balanced with minimal rebalancing during insertion and deletion operations.

### **Red-Black Tree Properties:**
1. **Each node is either red or black**.
2. **The root is always black**.
3. **Red nodes cannot have red children** (i.e., no two red nodes can be adjacent).
4. **Every path from a node to its descendant leaves must have the same number of black nodes**.
5. **Newly inserted nodes are always red**.

These properties help ensure that the tree remains balanced, which leads to efficient performance for the basic tree operations.

### **Java Implementation of a Red-Black Tree**

The implementation includes the following operations:
1. **Insert**: Inserts a new node into the tree while ensuring that the Red-Black Tree properties are maintained.
2. **Left and Right Rotations**: To maintain balance during insertion and deletion.
3. **Fixing Violations**: After insertion, the tree might violate one or more of the Red-Black Tree properties, which are corrected through specific fixes (recoloring and rotations).

### 1. **Red-Black Tree Node Class**

```java
class Node {
    int data;
    Node parent, left, right;
    int color; // 1 for Red, 0 for Black

    // Constructor for the node
    public Node(int data) {
        this.data = data;
        this.parent = null;
        this.left = null;
        this.right = null;
        this.color = 1; // New nodes are always red
    }
}
```

### 2. **Red-Black Tree Class**

```java
class RedBlackTree {
    private Node root;
    private Node TNULL; // Sentinel node

    // Constructor to initialize the tree
    public RedBlackTree() {
        TNULL = new Node(0);
        TNULL.color = 0; // TNULL is always black
        root = TNULL;
    }

    // Left rotate operation
    private void leftRotate(Node x) {
        Node y = x.right;
        x.right = y.left;
        if (y.left != TNULL) {
            y.left.parent = x;
        }
        y.parent = x.parent;
        if (x.parent == null) {
            root = y;
        } else if (x == x.parent.left) {
            x.parent.left = y;
        } else {
            x.parent.right = y;
        }
        y.left = x;
        x.parent = y;
    }

    // Right rotate operation
    private void rightRotate(Node x) {
        Node y = x.left;
        x.left = y.right;
        if (y.right != TNULL) {
            y.right.parent = x;
        }
        y.parent = x.parent;
        if (x.parent == null) {
            root = y;
        } else if (x == x.parent.right) {
            x.parent.right = y;
        } else {
            x.parent.left = y;
        }
        y.right = x;
        x.parent = y;
    }

    // Fix the tree after insertion
    private void fixInsert(Node k) {
        Node u;
        while (k.parent.color == 1) {
            if (k.parent == k.parent.parent.right) {
                u = k.parent.parent.left;
                if (u.color == 1) {
                    u.color = 0;
                    k.parent.color = 0;
                    k.parent.parent.color = 1;
                    k = k.parent.parent;
                } else {
                    if (k == k.parent.left) {
                        k = k.parent;
                        rightRotate(k);
                    }
                    k.parent.color = 0;
                    k.parent.parent.color = 1;
                    leftRotate(k.parent.parent);
                }
            } else {
                u = k.parent.parent.right;
                if (u.color == 1) {
                    u.color = 0;
                    k.parent.color = 0;
                    k.parent.parent.color = 1;
                    k = k.parent.parent;
                } else {
                    if (k == k.parent.right) {
                        k = k.parent;
                        leftRotate(k);
                    }
                    k.parent.color = 0;
                    k.parent.parent.color = 1;
                    rightRotate(k.parent.parent);
                }
            }
            if (k == root) {
                break;
            }
        }
        root.color = 0;
    }

    // Insert a new node
    public void insert(int key) {
        Node node = new Node(key);
        node.parent = null;
        node.data = key;
        node.left = TNULL;
        node.right = TNULL;
        node.color = 1; // New node is always red

        Node y = null;
        Node x = root;

        while (x != TNULL) {
            y = x;
            if (node.data < x.data) {
                x = x.left;
            } else {
                x = x.right;
            }
        }

        node.parent = y;
        if (y == null) {
            root = node;
        } else if (node.data < y.data) {
            y.left = node;
        } else {
            y.right = node;
        }

        if (node.parent == null) {
            node.color = 0;
            return;
        }

        if (node.parent.parent == null) {
            return;
        }

        fixInsert(node);
    }

    // In-order traversal
    public void inorder() {
        inorderHelper(this.root);
    }

    private void inorderHelper(Node node) {
        if (node != TNULL) {
            inorderHelper(node.left);
            System.out.print(node.data + " ");
            inorderHelper(node.right);
        }
    }

    // Search for a node
    public boolean search(int key) {
        return searchTreeHelper(this.root, key);
    }

    private boolean searchTreeHelper(Node node, int key) {
        if (node == TNULL) {
            return false;
        }
        if (key == node.data) {
            return true;
        }

        if (key < node.data) {
            return searchTreeHelper(node.left, key);
        }
        return searchTreeHelper(node.right, key);
    }
}
```

### 3. **Main Class to Demonstrate Red-Black Tree Operations**

```java
public class Main {
    public static void main(String[] args) {
        RedBlackTree tree = new RedBlackTree();

        // Insert nodes into the Red-Black Tree
        tree.insert(10);
        tree.insert(20);
        tree.insert(30);
        tree.insert(15);
        tree.insert(25);
        tree.insert(5);

        // Display the tree in in-order (sorted) traversal
        System.out.println("In-order traversal of the Red-Black Tree:");
        tree.inorder();  // Output: 5 10 15 20 25 30
        System.out.println();

        // Searching for a value
        System.out.println("Is 15 in the tree? " + tree.search(15));  // Output: true
        System.out.println("Is 40 in the tree? " + tree.search(40));  // Output: false
    }
}
```

### **Explanation:**
1. **Node Class**: Represents each node in the tree. It includes data, parent, left, and right pointers. It also has a `color` field where `1` represents red and `0` represents black.
2. **RedBlackTree Class**:
   - **insert()**: Inserts a node into the tree, ensuring the Red-Black Tree properties are maintained. It uses the `fixInsert()` function to correct any violations after insertion.
   - **fixInsert()**: Ensures that the tree properties are maintained after insertion by performing appropriate rotations and recoloring.
   - **rotations**: Left and right rotations are used to maintain balance.
   - **inorder()**: Displays the tree in sorted order using in-order traversal.
   - **search()**: Searches for a specific value in the tree.
3. **Main Class**: Demonstrates how to insert nodes into the tree, perform a traversal, and search for a value.

### **Sample Output:**

```
In-order traversal of the Red-Black Tree:
5 10 15 20 25 30 

Is 15 in the tree? true
Is 40 in the tree? false
```

### **Time Complexity:**
- **Insert**: O(log n) – The height of the tree is kept balanced, ensuring efficient insertions.
- **Search**: O(log n) – Searching is efficient due to the balanced structure.
- **Traversal**: O(n) – Traversing the entire tree takes linear time.

### **Rotations Overview:**
- **Left Rotation**: Used when the right subtree of a node becomes too heavy (right-right or right-left case).
- **Right Rotation**: Used when the left subtree of a node becomes too heavy (left-left or left-right case).

These rotations are key to maintaining the balance in the Red-Black Tree, which ensures that all operations (insertion, deletion, and search) remain efficient.