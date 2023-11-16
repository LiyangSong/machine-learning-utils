# machine-learning-utils

This repo would store commonly used functions in machine learning, such that the work process in jupyter notebook can be clean and readable. 

# how to use

### Method 1:

- Clone this Repo by `git clone`.
- Import local `.py` file as packages in `.jypnb` file.

### Method 2:

- Click `.py` file and click `Raw`.
- Copy the url.
- In `.jypnb` file, using `requests.get` method to get the content of url.
- Write the `response.text` as local `.py` file.
- Import created local `.py` file as packages.
