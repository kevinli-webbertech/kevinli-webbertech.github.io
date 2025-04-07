# Tree

## Level order traversal

### Basic Level Order Implementation

**BFS** can be used to traverse a tree level wise. One key point to note is that we can control which way we want to traverse
(either left to right or right to left) by deciding what goes inside queue first.

**DFS** can also be used to solve these type of problems by adding a level variable.

102 Binary Tree Level Order Traversal

107 Binary Tree Bottom up Level Order Traversal

429 N-ary Tree Level Order Traversal

637 Average of Levels in Binary Tree

103 Binary Tree Zigzag Level Order Traversal

623 Add One Row at a given Level in a binary Tree

2415 Reverse Odd Levels of Binary Tree

2471 Minimum Number of Operations to Sort a Binary Tree by Level - CyclicSort+BFS

##  Depth/Height based

Note:Always check height is 0 or 1 indexed.

104 Maximum Depth of Binary Tree - Classic DFS

111 Minimum Depth of Binary Tree - BFS intuitive

559 Maximum Depth of N-ary Tree

## Counting

* Count nodes with some condition
* Count Trees or subtrees

https://leetcode.com/problems/count-complete-tree-nodes/
https://leetcode.com/problems/count-good-nodes-in-binary-tree/
https://leetcode.com/problems/count-nodes-equal-to-average-of-subtree/

250 Count Univalue Subtrees [Premium] [CN]

## Comparison on Two Trees

In these types of problem we generally run dfs/bfs simultaneously on both trees

100 Same Tree

101 Symmetric Tree

872 Leaf-Similar Trees

572 Subtree of Another Tree

## Path

Total number of paths between any two nodes in a binary tree with n nodes is:
n*(n−1)/2

For each starting node, there are n−1 other nodes in the tree to which it can connect, forming n−1 paths.

687 Longest Univalue Path

124 Binary Tree Maximum Path Sum H

### Rooted Path (Root to leaves)

When a path starts from root and ends at leaf node

1022 Sum of Root To Leaf Binary Numbers path forms a binary number

129 Sum Root to Leaf Numbers path forms a number

257 Binary Tree Paths Backtracking

988 Smallest String Starting From Leaf

112 Path Sum

### Tree Leaves [Try to get the question numbers here]

Operations done on leaf nodes

https://leetcode.com/problems/sum-of-left-leaves/
https://leetcode.com/problems/deepest-leaves-sum

## Ancestor

236 Lowest Common Ancestor of a Binary Tree✅
https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/
https://leetcode.com/problems/kth-ancestor-of-a-tree-node/
https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/

## Graph Creation

Use this option when you need more freedom to move inside a tree

863 All Nodes Distance K in Binary Tree better alternatives might be available


