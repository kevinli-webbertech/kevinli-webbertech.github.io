A **Binary Search Tree (BST)** is a type of binary tree that follows these properties:
1. **Left Subtree**: All nodes in the left subtree of a node have values less than the node's value.
2. **Right Subtree**: All nodes in the right subtree of a node have values greater than the node's value.
3. **Unique Elements**: The tree does not allow duplicate values.

BSTs allow efficient operations like **search**, **insert**, and **delete** because they have an ordered structure. Searching, insertion, and deletion typically take O(log n) time in a balanced BST, but can degrade to O(n) if the tree becomes unbalanced.

### **Binary Search Tree (BST) Implementation in Java**

In this implementation, we'll include methods to:
1. **Insert**: Add a node to the tree.
2. **Search**: Find if a value exists in the tree.
3. **Delete**: Remove a node from the tree.
4. **In-order Traversal**: Print nodes in ascending order.
5. **Pre-order and Post-order Traversal**: Print nodes in different traversal orders.

### 1. **BST Node Class**

```java
class Node {
    int data;   // Data in the node
    Node left;  // Left child
    Node right; // Right child

    // Constructor to create a new node
    public Node(int data) {
        this.data = data;
        this.left = null;
        this.right = null;
    }
}
```

### 2. **Binary Search Tree Class**

```java
class BinarySearchTree {
    Node root; // Root node of the binary search tree

    // Constructor to initialize the BST
    public BinarySearchTree() {
        root = null; // Initially, the tree is empty
    }

    // Insert a node with the given data
    public void insert(int data) {
        root = insertRec(root, data);
    }

    // Recursive function to insert a new node
    private Node insertRec(Node root, int data) {
        // If the tree is empty, create a new node
        if (root == null) {
            root = new Node(data);
            return root;
        }

        // Otherwise, recur down the tree
        if (data < root.data) {
            root.left = insertRec(root.left, data);  // Insert in the left subtree
        } else if (data > root.data) {
            root.right = insertRec(root.right, data);  // Insert in the right subtree
        }

        // Return the unchanged root node
        return root;
    }

    // Search for a node with the given key
    public boolean search(int key) {
        return searchRec(root, key);
    }

    // Recursive function to search for a node with the given key
    private boolean searchRec(Node root, int key) {
        // Base case: root is null or key is present at the root
        if (root == null) {
            return false;
        }
        if (root.data == key) {
            return true;
        }

        // Key is greater than the root's data
        if (key > root.data) {
            return searchRec(root.right, key);
        }

        // Key is smaller than the root's data
        return searchRec(root.left, key);
    }

    // Delete a node with the given key
    public void delete(int key) {
        root = deleteRec(root, key);
    }

    // Recursive function to delete a node with the given key
    private Node deleteRec(Node root, int key) {
        // Base case: root is null
        if (root == null) {
            return root;
        }

        // Traverse the tree to find the node to be deleted
        if (key < root.data) {
            root.left = deleteRec(root.left, key); // Key is smaller, search in left subtree
        } else if (key > root.data) {
            root.right = deleteRec(root.right, key); // Key is greater, search in right subtree
        } else {
            // Node to be deleted is found
            // Case 1: Node has no child (leaf node)
            if (root.left == null && root.right == null) {
                return null;
            }
            // Case 2: Node has one child
            if (root.left == null) {
                return root.right;
            } else if (root.right == null) {
                return root.left;
            }

            // Case 3: Node has two children
            root.data = minValue(root.right);  // Get the inorder successor
            root.right = deleteRec(root.right, root.data); // Delete the inorder successor
        }

        return root;
    }

    // Find the minimum value node in the tree (used for deletion)
    private int minValue(Node root) {
        int minValue = root.data;
        while (root.left != null) {
            minValue = root.left.data;
            root = root.left;
        }
        return minValue;
    }

    // In-order traversal of the BST (Left, Root, Right)
    public void inorder() {
        inorderRec(root);
    }

    private void inorderRec(Node root) {
        if (root != null) {
            inorderRec(root.left);  // Traverse left subtree
            System.out.print(root.data + " ");  // Visit the root
            inorderRec(root.right); // Traverse right subtree
        }
    }

    // Pre-order traversal of the BST (Root, Left, Right)
    public void preorder() {
        preorderRec(root);
    }

    private void preorderRec(Node root) {
        if (root != null) {
            System.out.print(root.data + " ");  // Visit the root
            preorderRec(root.left);  // Traverse left subtree
            preorderRec(root.right); // Traverse right subtree
        }
    }

    // Post-order traversal of the BST (Left, Right, Root)
    public void postorder() {
        postorderRec(root);
    }

    private void postorderRec(Node root) {
        if (root != null) {
            postorderRec(root.left);  // Traverse left subtree
            postorderRec(root.right); // Traverse right subtree
            System.out.print(root.data + " ");  // Visit the root
        }
    }
}
```

### 3. **Main Class to Demonstrate BST Operations**

```java
public class Main {
    public static void main(String[] args) {
        BinarySearchTree tree = new BinarySearchTree();

        // Inserting nodes into the BST
        tree.insert(50);
        tree.insert(30);
        tree.insert(20);
        tree.insert(40);
        tree.insert(70);
        tree.insert(60);
        tree.insert(80);

        // Display the tree using different traversals
        System.out.println("In-order traversal:");
        tree.inorder();  // Output: 20 30 40 50 60 70 80
        System.out.println();

        System.out.println("Pre-order traversal:");
        tree.preorder();  // Output: 50 30 20 40 70 60 80
        System.out.println();

        System.out.println("Post-order traversal:");
        tree.postorder();  // Output: 20 40 30 60 80 70 50
        System.out.println();

        // Searching for a value in the binary search tree
        int key = 60;
        System.out.println("Is " + key + " in the tree? " + tree.search(key));  // Output: true

        key = 100;
        System.out.println("Is " + key + " in the tree? " + tree.search(key));  // Output: false

        // Deleting a node from the tree
        System.out.println("Deleting node with value 20");
        tree.delete(20);

        // Display the tree after deletion
        System.out.println("In-order traversal after deletion:");
        tree.inorder();  // Output: 30 40 50 60 70 80
    }
}
```

### **Explanation:**
- **Node Class**: Represents each node in the tree. Each node contains data and references to its left and right children.
- **BinarySearchTree Class**:
  - **insert()**: Adds a new node to the tree while maintaining the BST property.
  - **search()**: Searches for a value in the BST.
  - **delete()**: Removes a node from the tree and handles three cases: no children, one child, or two children.
  - **inorder(), preorder(), postorder()**: Perform different types of tree traversal.
- **Main Class**: Demonstrates the creation of a BST, insertion, deletion, searching, and traversal operations.

### **Sample Output:**

```
In-order traversal:
20 30 40 50 60 70 80 

Pre-order traversal:
50 30 20 40 70 60 80 

Post-order traversal:
20 40 30 60 80 70 50 

Is 60 in the tree? true
Is 100 in the tree? false
Deleting node with value 20
In-order traversal after deletion:
30 40 50 60 70 80 
```

### **Time Complexity:**
- **insert()**: O(h) – Where h is the height of the tree. In the worst case (for an unbalanced tree), this can be O(n), where n is the number of nodes.
- **search()**: O(h) – Searching takes time proportional to the height of the tree.
- **delete()**: O(h) – Deletion also takes time proportional to the height of the tree.
- **Traversal (inorder, preorder, postorder)**: O(n) – All tree traversal operations visit every node once.

### **Balanced vs. Unbalanced BST**:
- If the tree becomes unbalanced, the height can grow to O(n), which means operations can degrade to O(n).
- A **Balanced Binary Search Tree (BST)** (such as an **AVL Tree** or **Red-Black Tree**) ensures that the tree remains balanced, keeping the height logarithmic (O(log n)) and ensuring that operations remain efficient.

Let me know if you'd like to explore **AVL trees**, **Red-Black trees**, or other advanced topics!