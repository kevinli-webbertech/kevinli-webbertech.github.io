# Keras Installation Guide

This guide provides step-by-step instructions to install Keras on any operating system with Python 3.5 and above.

## Prerequisites

- Python 3.5 and above installed on your system
- pip (Python package installer)

## Steps for Installation

### Step 1: Create a Virtual Environment

Using a virtual environment to manage Python packages for different projects is recommended to avoid conflicts between packages installed in different environments.

#### Linux/Mac OS

1. Open your terminal.
2. Navigate to your project root directory.
3. Create a virtual environment by running the following command:

    ```bash
    python3 -m venv kerasenv
    ```

    This command creates a directory named `kerasenv` with `bin`, `lib`, and `include` folders.

4. Activate the virtual environment:

    ```bash
    source kerasenv/bin/activate
    ```

#### Windows

1. Open your command prompt.
2. Navigate to your project root directory.
3. Create a virtual environment by running the following command:

    ```bash
    python -m venv kerasenv
    ```

    This command creates a directory named `kerasenv` with `Scripts`, `Lib`, and `Include` folders.

4. Activate the virtual environment:

    ```bash
    kerasenv\Scripts\activate
    ```

### Step 2: Upgrade pip

Before installing Keras, upgrade pip to the latest version. Run the following command:

```bash
pip install --upgrade pip


Windows

-Go to “kerasvenv” and give the command below

.\env\Scripts\activate
```

### Python Libraries

To work with the project, ensure the following Python libraries are installed:

- **Numpy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Scipy**
- **Seaborn**


### Step 5: Install Required Python Libraries

To ensure you have all necessary libraries for working with Keras, install the following packages:

- **Numpy**

    ```bash
    pip install numpy
    ```

- **Pandas**

    ```bash
    pip install pandas
    ```

- **Scikit-learn**

    ```bash
    pip install scikit-learn
    ```

- **Matplotlib**

    ```bash
    pip install matplotlib
    ```

- **Scipy**

    ```bash
    pip install scipy
    ```

- **Seaborn**

    ```bash
    pip install seaborn
    ```


# Keras Installation Using Python

To install Keras, use the following pip command:

```bash
pip install keras

## Quit Virtual Environment

After making changes in your project, you can exit the virtual environment by using the following command:

```bash
deactivate

```
# Anaconda Cloud

We believe that you have installed Anaconda Cloud on your machine. If Anaconda is not installed, then visit the official link, [Anaconda Download](https://www.anaconda.com/download) and choose the download based on your OS.

## Create a New Conda Environment

```bash
conda create --name PythonCPU

```
## Activate Conda Environment

```bash
activate PythonCPU
```
## Install Spyder

```bash
conda install spyder
```

## Install Keras

```bash
conda install -c anaconda keras
```

## Launch Spyder

```bash
spyder
```

## Verify Installation

To ensure everything was installed correctly, import all the modules. If anything went wrong, you will get a "module not found" error message.





