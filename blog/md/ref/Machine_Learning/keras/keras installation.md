'KERAS INSTALLATION'

-ANY KIND OF OS
-Python--3.5 and above

steps for installation

1.create virtual environment

Virtualenv-to manage python packages for different projects
recommended to avoid breaking packages installed in other environments.

Linux/Mac OS

go to project root directory 
 
-type the command-python3 -m venv kerasenv

“kerasenv” directory is created with bin,lib and include folders upon executing the above command.

Windows

use command-py -m venv keras

2.Activate the environment

Linux/Mac OS

Go to “kerasvenv” and give the command below

-$ cd kerasvenv kerasvenv $ source bin/activate


Windows

-Go to “kerasvenv” and give the command below

.\env\Scripts\activate

3.Python libraries
Numpy
Pandas
Scikit-learn
Matplotlib
Scipy
Seaborn

install all the above libraries

numpy

pip install numpy 

following response

Collecting numpy 
   Downloading 
https://files.pythonhosted.org/packages/cf/a4/d5387a74204542a60ad1baa84cd2d3353c330e59be8cf2d47c0b11d3cde8/ 
   numpy-3.1.1-cp36-cp36m-macosx_10_6_intel.
macosx_10_9_intel.macosx_10_9_x86_64. 
   macosx_10_10_intel.macosx_10_10_x86_64.whl (14.4MB) 
      |████████████████████████████████| 14.4MB 2.8MB/s
pandas

pip install pandas

following response

Collecting pandas 
   Downloading 
https://files.pythonhosted.org/packages/cf/a4/d5387a74204542a60ad1baa84cd2d3353c330e59be8cf2d47c0b11d3cde8/ 
pandas-3.1.1-cp36-cp36m-macosx_10_6_intel.
macosx_10_9_intel.macosx_10_9_x86_64. 
   macosx_10_10_intel.macosx_10_10_x86_64.whl (14.4MB) 
      |████████████████████████████████| 14.4MB 2.8MB/s


matplotlib

pip install matplotlib

following response

Collecting matplotlib 
   Downloading 
https://files.pythonhosted.org/packages/cf/a4/d5387a74204542a60ad1baa84cd2d3353c330e59be8cf2d47c0b11d3cde8/ 
matplotlib-3.1.1-cp36-cp36m-macosx_10_6_intel.
macosx_10_9_intel.macosx_10_9_x86_64. 
   macosx_10_10_intel.macosx_10_10_x86_64.whl (14.4MB) 
      |████████████████████████████████| 14.4MB 2.8MB/s

scipy

pip install scipy

following response,

Collecting scipy 
   Downloading 
https://files.pythonhosted.org/packages/cf/a4/d5387a74204542a60ad1baa84cd2d3353c330e59be8cf2d47c0b11d3cde8 
/scipy-3.1.1-cp36-cp36m-macosx_10_6_intel.
macosx_10_9_intel.macosx_10_9_x86_64. 
   macosx_10_10_intel.macosx_10_10_x86_64.whl (14.4MB) 
      |████████████████████████████████| 14.4MB 2.8MB/s


scikit-learn

machine learning library used for classification, regression and clustering algorithms
requirements for installation

Python version 3.5 or higher
NumPy version 1.11.0 or higher
SciPy version 0.17.0 or higher
joblib 0.11 or higher.

command-pip install -U scikit-learn

Seaborn
-allows to visualize data
command-pip install seaborn

following response

Collecting seaborn 
   Downloading 
https://files.pythonhosted.org/packages/a8/76/220ba4420459d9c4c9c9587c6ce607bf56c25b3d3d2de62056efe482dadc 
/seaborn-0.9.0-py3-none-any.whl (208kB) 100% 
   |████████████████████████████████| 215kB 4.0MB/s 
Requirement already satisfied: numpy> = 1.9.3 in 
./lib/python3.7/site-packages (from seaborn) (1.17.0) 
Collecting pandas> = 0.15.2 (from seaborn) 
   Downloading 
https://files.pythonhosted.org/packages/39/b7/441375a152f3f9929ff8bc2915218ff1a063a59d7137ae0546db616749f9/ 
pandas-0.25.0-cp37-cp37m-macosx_10_9_x86_64.
macosx_10_10_x86_64.whl (10.1MB) 100% 
   |████████████████████████████████| 10.1MB 1.8MB/s 
Requirement already satisfied: scipy>=0.14.0 in 
./lib/python3.7/site-packages (from seaborn) (1.3.0) 
Collecting matplotlib> = 1.4.3 (from seaborn) 
   Downloading 
https://files.pythonhosted.org/packages/c3/8b/af9e0984f
5c0df06d3fab0bf396eb09cbf05f8452de4e9502b182f59c33b/ 
matplotlib-3.1.1-cp37-cp37m-macosx_10_6_intel.
macosx_10_9_intel.macosx_10_9_x86_64 
.macosx_10_10_intel.macosx_10_10_x86_64.whl (14.4MB) 100% 
   |████████████████████████████████| 14.4MB 1.4MB/s 
...................................... 
...................................... 
Successfully installed cycler-0.10.0 kiwisolver-1.1.0 
matplotlib-3.1.1 pandas-0.25.0 pyparsing-2.4.2 
python-dateutil-2.8.0 pytz-2019.2 seaborn-0.9.0


Keras Installation Using Python

pip install keras

Quit virtual environment

after changes in the project enter command - deactivate


Anaconda Cloud

We believe that you have installed anaconda cloud on your machine. If anaconda is not installed, then visit the official link, https://www.anaconda.com/download and choose download based on your OS.


Create a new conda environment

-conda create --name PythonCPU

Activate conda environment

-activate PythonCPU

Install spyder

-conda install spyder

Install Keras

-conda install -c anaconda keras

Launch spyder

-spyder

To ensure everything was installed correctly, import all the modules, it will add everything and if anything went wrong, you will get module not found error message.




