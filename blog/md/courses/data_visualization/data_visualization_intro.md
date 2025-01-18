# Introduction Data Visualization

https://en.wikipedia.org/wiki/Data_and_information_visualization

## Graphics and Traditional Visualization

![alt text](image-6.png)

## Visualization

## Visualize Data from Data Analysis

### Matplotlib

Matplotlib is a Python plotting library that provides a MATLAB-like interface. Here are a few examples of how to create plots using Matplotlib that resemble MATLAB plots:

Line Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.grid(True)
plt.show()
```

Scatter Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
x = np.random.rand(100)
y = np.random.rand(100)

# Create the plot
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.grid(True)
plt.show()
```

Bar Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Create some data
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]

# Create the plot
plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.grid(True)
plt.show()
```

Histogram

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
data = np.random.randn(1000)

# Create the plot
plt.hist(data, bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)
plt.show()
```

![matplotlib](matplotlib.png)

![matplotlib1](matplotlib1.png)

![matplotlib1](matplotlib2.png)

### Seaborn

![seaborn](seaborn.png)

![seaborn1](seaborn1.png)

![seaborn2](seaborn2.png)

## D3 - A JavaScript Visualization Library for the Web Development

![d3](d3.png)

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

## Apache Echarts

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)

## Visualization is everywhere

![alt text](image-7.png)

![alt text](image-8.png)

![alt text](image-9.png)

![alt text](image-10.png)

![alt text](image-11.png)


Snowflake uses the above programming language's libraries to visualize data.

![alt text](image-12.png)

![alt text](image-13.png)

![alt text](image-14.png)

![alt text](image-15.png)

## Visualization in Business Application

* Tableau

* PowerBI

Since we are business major, we are more interested in learning more specialized softwares such as Tableau and PowerBI.
In the next few classes we will be learning a little more about these two softwares.
