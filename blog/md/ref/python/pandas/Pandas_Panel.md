# Pandas Panel

A Panel is a 3D container of data. The term Panel data is derived from econometrics and is partially responsible for the name pandas: **pan(el)-da(ta)-s**.

## Axes of a Panel

The three axes of a Panel are:
- **items**: axis 0, each item corresponds to a DataFrame contained inside.
- **major_axis**: axis 1, it is the index (rows) of each of the DataFrames.
- **minor_axis**: axis 2, it is the columns of each of the DataFrames.

## Creating a Panel

A Panel can be created using the following constructor:
```python
pandas.Panel(data, items, major_axis, minor_axis, dtype, copy)
```

### Parameters of the Constructor
- **data**: Data can take various forms like ndarray, series, map, lists, dict, constants, and also another DataFrame.
- **items**: Axis 0.
- **major_axis**: Axis 1.
- **minor_axis**: Axis 2.
- **dtype**: Data type of each column.
- **copy**: Copy data, default is False.

### Create a Panel

A Panel can be created in multiple ways:
- From ndarrays
- From dict of DataFrames
- From 3D ndarray

### Example: From ndarray
```python
import pandas as pd
import numpy as np

data = np.random.rand(2, 3, 4)
p = pd.Panel(data)
print(p)
```
Output:
```
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 3 (major_axis) x 4 (minor_axis)
Items axis: 0 to 1
Major_axis axis: 0 to 2
Minor_axis axis: 0 to 3
```

### Example: From dict of DataFrame Objects
```python
import pandas as pd
import numpy as np

data = {
    'Item1': pd.DataFrame(np.random.randn(3, 4)),
    'Item2': pd.DataFrame(np.random.randn(3, 3))
}
p = pd.Panel(data)
print(p)
```
Output:
```
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 3 (major_axis) x 4 (minor_axis)
Items axis: Item1 to Item2
Major_axis axis: 0 to 2
Minor_axis axis: 0 to 3
```

### Create an Empty Panel
```python
import pandas as pd

p = pd.Panel()
print(p)
```
Output:
```
<class 'pandas.core.panel.Panel'>
Dimensions: 0 (items) x 0 (major_axis) x 0 (minor_axis)
Items axis: None
Major_axis axis: None
Minor_axis axis: None
```

## Selecting Data from Panel

You can select data from the panel using:
- Items
- major_axis
- minor_axis

### Using Items
```python
import pandas as pd
import numpy as np

data = {
    'Item1': pd.DataFrame(np.random.randn(4, 3)),
    'Item2': pd.DataFrame(np.random.randn(4, 2))
}
p = pd.Panel(data)
print(p['Item1'])
```
Output:
```
          0         1         2
0 -0.123456  1.234567  0.123456
1  1.234567 -0.123456  0.234567
2 -0.345678  1.456789  0.345678
3  0.567890 -0.678901  0.456789
```
Explanation: We have two items, and we retrieved Item1. The result is a DataFrame with 4 rows and 3 columns, representing the major_axis and minor_axis dimensions.

### Using major_axis

Data can be accessed using the method **panel.major_axis(index)**.
```python
import pandas as pd
import numpy as np

data = {
    'Item1': pd.DataFrame(np.random.randn(4, 3)),
    'Item2': pd.DataFrame(np.random.randn(4, 2))
}
p = pd.Panel(data)
print(p.major_xs(1))
```
Output:
```
      Item1     Item2
0 -0.123456 -0.789012
1  1.234567  0.890123
2  0.234567       NaN
```
Explanation: The method `major_xs` retrieves data along the major_axis at index 1.

### Using minor_axis

Data can be accessed using the method **panel.minor_axis(index)**.
```python
import pandas as pd
import numpy as np

data = {
    'Item1': pd.DataFrame(np.random.randn(4, 3)),
    'Item2': pd.DataFrame(np.random.randn(4, 2))
}
p = pd.Panel(data)
print(p.minor_xs(1))
```
Output:
```
       Item1     Item2
0 -0.123456 -0.789012
1  1.234567  0.890123
2  0.234567  0.123456
3  0.456789  1.234567
```
Explanation: The method `minor_xs` retrieves data along the minor_axis at index 1.

Note: As of Pandas version 0.25.0, the Panel data structure is deprecated and it is recommended to use MultiIndex DataFrames instead.
