# Machine Learning

### Processing in Machine Learning

- Data Proprocessing
- Modeling Data
- Evaluation

### Feature Scaling

Scaling guarantees that all features are on a comparable scale and have comparable ranges.

1. Normalization - scale the range in between [0,1]

```
x' = x - min(x) / max(x) - min(x)
```

2. Standardization

```
x' = x - avg(x) / deviation(x)
```

### Flow of Machine Learning
1) Data Proproccessing
    - Import Data
    - Clean the data
    - Split the data into training set and test sets
2) Modelling
    - Build
    - Train
    - Make Prediction
3) Evaluation
    - Calculate Performance metrics
    - Make a verdict

 - Traning set and Test set
### Feature Scaling
    Applied to columns
    #### What?
        Normalizing range of features
        1) Normalization
            X`= (X - min(X))/(max(X)-min(X)) 
            result = [0, 1]
        2) Standardization
            X`= X-avg(X)/deviation(X)
            result = [-3,3]
    #### why?
        features with different degress of magnitude, range and units
        Interpret these features in same scale





### Data Pre-Processing
- Importing Liabraries
- Importing Dataset
- Missing data handling
- Encoding data
- Splitting data
- Feature Scaling
#### Importing Liabraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
        1) numpy - allow working with array
        2) matplotlib - charts
        3) pandas - import and mertics features
#### Importing Dataset
```python
import pandas as pd
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values
```

#### Taking care of the missing data
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
```
#### Encoding categorical data and Encoding the independent Variable
```python

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Implement an instance of the ColumnTransformer class
categorical_features = ['Sex','Pclass','Embarked']
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),categorical_features)],remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(df)

X = np.array(X)

# Print the updated matrix of features and the dependent variable vector
print(X)

```
#### Encoding the Dependent Variable
```python
from sklearn.preprocessing import LabelEncoder
# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
Y = le.fit_transform(df['Survived'])
```
    - Splitting the dataset into the traning set and test set
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
```
#### Feature Scaling

### numpy (Array Operations)

Package for Scientific computation and data manipulation
support for large, multi-dimensional arrays and matrices
`mathematical functions`, `array creation`, `linear algebra`, `array manipulation`, `numerical computation`
`np,array`, `np.arange(0,10,2)`, `np,mean`, `np.dot(array1,array2)`,`np.linalg.inv(array1)`,`np.max(array1)`,`np.min(array1)`,`np.sum(array1)`,`np.vstack`,`np.hstack`

```python
import numpy as np

# Create a 1D array
arr1d = np.array([1, 2, 3, 4, 5])

# Create a 2D array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# Perform element-wise arithmetic operations
result = arr1d * 2
print(result)  # Output: [ 2  4  6  8 10]

# Generate a sequence of numbers
sequence = np.arange(0, 10, 2)
print(sequence)  # Output: [0 2 4 6 8]

# Compute the mean of an array
mean_value = np.mean(arr1d)
print(mean_value)  # Output: 3.0

```

### Pandas (data analysis, data cleaning, data preprocessing)
working with structured data
`data input and output`, `data cleaning and processing`, `data indexing and selection`, `aggregation and grouping`, `merging and joining`

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data into a DataFrame
data = pd.read_csv('data.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Summary statistics
print(data.describe())

# Create a histogram of a column
data['age'].hist()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

### matplotlib (data visualization)
tool for creating static, animated, and interactive visualization
`pyplot` matlab like interface for creating plots and visualization

Plot Types
`plt.plot()` - line , `plt.scatter()` - scatter, `plt.bar()` - bar, `plt.hist()` - histogram, `plt.pie()` - bar
`plt.subplot()`, `plt.annotation()`, `plt.text()`

save and import, animation, interactive plots

```python
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 15, 7]

# Create a line plot
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data Points')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')

# Add a legend
plt.legend()

# Show the plot
plt.show()

```

### sklearn
Open source liabrary 
Library for Python
Provides wide range of tools for machine learning tasks
Built on top of NumPy, SciPy, matplotlib, Scikit-Learn
`feature selection`, `pipeline construction`, `feature selection`, `model inspection`
#### SimpleImputer 
class of sklearn.impute = provides simple strategy for imputing mising values in datasets
`Impution` - Process of filling empty values

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate some example data
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
new_data = np.array([[5]])
predicted = model.predict(new_data)
print("Predicted:", predicted)

```

