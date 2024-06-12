# Python Pandas - IO Tools

Pandas provides powerful tools for reading and writing data from various file formats, including CSV, Excel, JSON, SQL databases, and more. These IO tools make it easy to work with different data sources in a Pandas DataFrame.

## Reading Data

### CSV Files

```python
import pandas as pd

# Reading from CSV
df = pd.read_csv('file.csv')
```

### Excel Files

```python
# Reading from Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
```

### JSON Files

```python
# Reading from JSON
df = pd.read_json('file.json')
```

### SQL Databases

```python
from sqlalchemy import create_engine

# Create a connection to the database
engine = create_engine('sqlite:///:memory:')

# Reading from SQL
df = pd.read_sql_query('SELECT * FROM table_name', engine)
```

## Writing Data

### CSV Files

```python
# Writing to CSV
df.to_csv('file.csv', index=False)
```

### Excel Files

```python
# Writing to Excel
df.to_excel('file.xlsx', index=False)
```

### JSON Files

```python
# Writing to JSON
df.to_json('file.json', orient='records')
```

### SQL Databases

```python
# Writing to SQL
df.to_sql('table_name', engine)
```

## Other Formats

Pandas also supports reading and writing data in many other formats, including HTML, HDF5, Parquet, Stata, SAS, and more. The `pd.read_` and `df.to_` functions can be used with various file extensions to handle these formats.

## Conclusion

Pandas IO tools provide a convenient way to read and write data from a wide range of file formats, making it easy to work with different data sources in a Pandas DataFrame. Whether it's CSV, Excel, JSON, SQL, or other formats, Pandas has you covered.

