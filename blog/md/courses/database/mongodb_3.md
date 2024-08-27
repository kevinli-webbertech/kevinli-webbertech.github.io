### MongoDB Programming

## Outline

* MongoDB Query Operators
* MongoDB Update Operators
* Aggregation Pipelines

## MongoDB Query Operators

There are many query operators that can be used to compare and reference document fields.

## Comparison

The following operators can be used in queries to compare values:

$eq: Values are equal
$ne: Values are not equal
$gt: Value is greater than another value
$gte: Value is greater than or equal to another value
$lt: Value is less than another value
$lte: Value is less than or equal to another value
$in: Value is matched within an array

### Logical
The following operators can logically compare multiple queries.

$and: Returns documents where both queries match
$or: Returns documents where either query matches
$nor: Returns documents where both queries fail to match
$not: Returns documents where the query does not match

### Evaluation
The following operators assist in evaluating documents.

$regex: Allows the use of regular expressions when evaluating field values
$text: Performs a text search
$where: Uses a JavaScript expression to match documents

## MongoDB Update Operators
There are many update operators that can be used during document updates.

### Fields
The following operators can be used to update fields:

$currentDate: Sets the field value to the current date
$inc: Increments the field value
$rename: Renames the field
$set: Sets the value of a field
$unset: Removes the field from the document

### Array
The following operators assist with updating arrays.

$addToSet: Adds distinct elements to an array
$pop: Removes the first or last element of an array
$pull: Removes all elements from an array that match the query
$push: Adds an element to an array

## Aggregation Pipelines
Aggregation Pipelines
Aggregation operations allow you to group, sort, perform calculations, analyze data, and much more.

Aggregation pipelines can have one or more "stages". The order of these stages are important. Each stage acts upon the results of the previous stage.

```shell
db.posts.aggregate([
  // Stage 1: Only find documents that have more than 1 like
  {
    $match: { likes: { $gt: 1 } }
  },
  // Stage 2: Group documents by category and sum each categories likes
  {
    $group: { _id: "$category", totalLikes: { $sum: "$likes" } }
  }
])
```

[TODO] Mongo vs RDMS
[TODO](https://www.w3schools.com/mongodb/mongodb_aggregations_group.php)

