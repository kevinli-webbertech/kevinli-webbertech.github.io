# Seaborn - Introduction

Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing 
attractive and informative statistical graphics. In this tutorial, we will explore the basics of Seaborn and learn how 
to create different types of visualizations. Each section includes examples and tips to help you build a strong foundation.

## **Getting Started with Seaborn**

For simplicity in development and to avoid setup issues, you can also use Google Colab, a cloud-based Python environment.
Colab comes with many libraries, including Seaborn, pre-installed. To start using Google Colab:

1. Go to Google Colab.
2. Create a new notebook.

Run the following code to verify that Seaborn is available:

```python
import seaborn as sns
print(sns.__version__)
```

![seaborn1](../../../images/data_visualization/seaborn1.png)

This ensures you have a ready-to-use environment without needing to install anything locally.

If its not present it can be installed using pip:

```shell
!pip install seaborn
```

![seaborn1](../../../images/data_visualization/seaborn2.png)


### **Importing Seaborn**

You need to import Seaborn and other essential libraries like NumPy and Pandas:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

NumPy is used for numerical operations such as generating arrays, performing calculations, or creating custom data 
for plots. Pandas is essential for loading, cleaning, and manipulating datasets, providing the structured data that 
Seaborn requires for its visualizations.

### **Loading Datasets**

Seaborn comes with several built-in datasets that you can use to practice. To explore the list of available datasets, 
you can use:

To explore available datasets, use:

`print(sns.get_dataset_names())`

![seaborn1](../../../images/data_visualization/seaborn3.png)

```python
# Load the built-in "tips" dataset

data = sns.load_dataset("tips")

# Display the first few rows
data.head()
```

![seaborn1](../../../images/data_visualization/seaborn4.png)

### **Basic Plotting with Seaborn**

Seaborn provides a simpler and more intuitive interface compared to Matplotlib for creating complex visualizations 
with minimal code. While Matplotlib is highly flexible, Seaborn excels in statistical data visualization and 
integrates seamlessly with Pandas DataFrames, making it easier to plot data directly from tabular formats.

### **Scatterplot**
![scatterplot1.png](png/scatterplot/scatterplot1.png)
![scatterplot2.png](png/scatterplot/scatterplot2.png)
![scatterplot3.png](png/scatterplot/scatterplot3.png)
![scatterplot4.png](png/scatterplot/scatterplot4.png)
![scatterplot5.png](png/scatterplot/scatterplot5.png)
![scatterplot6.png](png/scatterplot/scatterplot6.png)

### **Histogram**
![histogram1.png](png/histogram/histogram1.png)
![histogram2.png](png/histogram/histogram2.png)
![histogram3.png](png/histogram/histogram3.png)
![histogram4.png](png/histogram/histogram4.png)
![histogram5.png](png/histogram/histogram5.png)
![histogram6.png](png/histogram/histogram6.png)
![histogram7.png](png/histogram/histogram7.png)

### **Categorical Visualization**
![categorical1.png](png/categorical/categorical1.png)
![catplot.png](png/categorical/catplot.png)
![violinplot.png](png/categorical/violinplot.png)
![boxplot.png](png/categorical/boxplot.png)

### **Regression**
![regression1.png](png/regression/regression1.png)
![regression2.png](png/regression/regression2.png)
![regression3.png](png/regression/regression3.png)

## **Tips for Using Seaborn**

1. Start with Built-in Datasets: Use Seaborn’s built-in datasets like tips or iris to practice.
2. Explore the Documentation: The [Seaborn documentation](https://seaborn.pydata.org/) is a great resource for understanding functions and parameters.
3. Combine with Pandas: Use Pandas to preprocess your data before visualizing it with Seaborn.
4. Experiment with Parameters: Don’t hesitate to experiment with various parameters to find the best settings for your data.
5. Save Your Plots: Use plt.savefig("plot.png") to save your visualizations for reports or presentations.