the major mathematical concepts of machine learning which is important from Natural Language Processing point of view 
Vectors

Vector is an array of numbers either continuous or discrete and space consisted is called as vector space.
space dimensions are finite or infinte .
most of the machine learning and datascience problems deal with fixed length vectors

vector representation as below
temp = torch.FloatTensor([23,24,24.5,26,27.2,23.0])
temp.size()
Output - torch.Size([6])

scalar
scalar is zero dimension containing only one value
in pytorch ,there is no special tensor
 scalar representation
x = torch.rand(10)
x.size()
Output - torch.Size([10])

Matrices
structured data is usually represented in the form of tables or a specific matrix
Boston House Prices available in python scikit learn library
boston_tensor = torch.from_numpy(boston.data)
boston_tensor.size()
Output: torch.Size([506, 13])
boston_tensor[:2]
Output:
Columns 0 to 7
0.0063 18.0000 2.3100 0.0000 0.5380 6.5750 65.2000 4.0900
0.0273 0.0000 7.0700 0.0000 0.4690 6.4210 78.9000 4.9671
Columns 8 to 12
1.0000 296.0000 15.3000 396.9000 4.9800
2.0000 242.0000 17.8000 396.9000 9.1400

