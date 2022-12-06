BINARY CLASSIFICATION
=

CHURN PREDICTION
-

$ g{(x_i)} ≈ y_i $

Target variable -> $y_i$
Feature vector describing the ith customer -> $x_i$

$y_i \in \{0, 1\}$

1 -> positive(Churn)
0 -> negative(Not Churn)

LIKELIHOOD OF CHURN
-
$ g(x_i)$ output is 0, 1

|CUSTOMER|a|b|c|d|e|<-__X__|
|--------|-|-|-|-|-|-|
|Churn/Not Churn|0|0|1|0|1|<-__y__|

Data preparation
-

- Download data
- View data
- Make column names and values look uniform

```py
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')
```
- Check if churn variables need preparation
    - Convert totalcharges from object to number and fill in missing values
    
    ```py
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)
    ```

    - Converted churn values `yes/no` to binary `0/1`
    
    ```py
    df.churn = (df.churn == 'yes').astype(int)
    ```
    
Feature importance
=

1. Difference
-

`GLOBAL - GROUP`

- `<0` - More likely to churn
- `>0` - Less likely to churn

2. Risk Ratio

RISK = $\frac{GROUP}{GLOBAL}$

- `<1` - More likely to churn
- `>1` - Less likely to churn

Feture Importance(def)
-

Feature Importance refers to techniques that calculate a score for all the input features for a given model — the scores simply represent the “importance” of each feature. A higher score means that the specific feature will have a larger effect on the model that is being used to predict a certain variable.

Why is Feature Importance so Useful?
Feature Importance is extremely useful for the following reasons:

1. Data Understanding.
Building a model is one thing, but understanding the data that goes into the model is another. Like a correlation matrix, feature importance allows you to understand the relationship between the features and the target variable. It also helps you understand what features are irrelevant for the model.

2. Model Improvement.
When training your model, you can use the scores calculated from feature importance to reduce the dimensionality of the model. The higher scores are usually kept and the lower scores are deleted as they are not important for the model. This not only makes the model simpler but also speeds up the model’s working, ultimately improving the performance of the model.

3. Model Interpretability.
Feature Importance is also useful for interpreting and communicating your model to other stakeholders. By calculating scores for each feature, you can determine which features attribute the most to the predictive power of your model.

Math Behind Feature Importance(Gini importance and Permutation feature importance.)
-

Gini Importance
-

In the Scikit-learn, Gini importance is used to calculate the node impurity and feature importance is basically a reduction in the impurity of a node weighted by the number of samples that are reaching that node from the total number of samples. This is known as node probability. Let us suppose we have a tree with two child nodes, the equation we have is:

$ n_{ij} = w_jC_j - w_{left(j)}C_{left(j)} - w_{right(j)}C_{right(j)} $

Here we have:

- $n_{ij}$ = node j importance
- $w_j$ = weighted number of samples reaching node j
- $C_j$ = the impurity value of node j
- $left_{(j)}$ = child node on left of node j
- $right_{(j)}$ = child node on right of node j

This equation gives us the importance of a node j which is used to calculate the feature importance for every decision tree. A single feature can be used in the different branches of the tree. Thus, we calculate the feature importance as follows.

$$
fi_i = \frac{\sum_j:{node \ j  \ splits \ on \ feature \ i} \ n_{ij}}{\sum_j \in {all \ nodes} \ n_{ij}}
$$

The features are normalized against the sum of all feature values present in the tree and after dividing it with the total number of trees in our random forest, we get the overall feature importance. With this, you can get a better grasp of the feature importance in random forests.

Permutation Feature Importance
-

The idea behind permutation feature importance is simple. The feature importance is calculated by noticing the increase or decrease in error when we permute the values of a feature. If permuting the values causes a huge change in the error, it means the feature is important for our model. The best thing about this method is that it can be applied to every machine learning model. Its approach is model agnostic which gives you a lot of freedom. There are no complex mathematical formulas behind it. The permutation feature importance is based on an algorithm that works as follows.
<br>

1. Calculate the mean squared error with the original values
2. Shuffle the values for the features and make predictions
3. Calculate the mean squared error with the shuffled values
4. Compare the difference between them
5. Sort the differences in descending order to get features with most to least importance


Feature Importance in Python
-

In this section, we’ll create a random forest model using the [Boston dataset](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset).
<br>
First, we’ll import all the required libraries and our dataset.

```py
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
```

The next step is to load the dataset and split it into a test and training set.

```py

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

Next, we’ll create the random forest model.

```pg
rf = RandomForestRegressor(n_estimators=150)
rf.fit(X_train, y_train)
```

Once the model is created, we can conduct feature importance and plot it on a graph to interpret the results easily.

```py
sort = rf.feature_importances_.argsort()
plt.barh(boston.feature_names[sort], rf.feature_importances_[sort])
plt.xlabel("Feature Importance")
```

RM is the average number of rooms per dwelling and it can be seen above that it is the most important feature in predicting the target variable.

Feature Importance with Gradio
-

Gradio is a beautiful package that helps create simple and interactive interfaces for machine learning models. With Gradio, you can evaluate and test your model in real time. An interesting thing about Gradio is that it calculates the feature importance with a single parameter and we can interact with the features to see how it affects feature importance.
<br>
Here’s an example:
<br>
First, we’ll import all the required libraries and our dataset. In this example, I will be using the [iris dataset](https://www.kaggle.com/datasets/vikrishnan/iris-dataset) from the Seaborn library.

Then we’ll split the dataset and fit it on the model.

```py
from sklearn.model_selection import train_test_split
X=iris.drop("species",axis=1)
y=iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.svm import SVC
model = SVC(probability=True)
model.fit(X_train,y_train)
```

We’ll also create a prediction function that will be used in our Gradio interface.

```
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
   df = pd.DataFrame.from_dict({'Sepal Length':[sepal_length],
                                'Sepal Width': [sepal_width],
                                'Petal Length': [petal_length],  
                                'Petal Width': [petal_width]})
   predict = model.predict_proba(df)[0]
   return {model.classes_[i]: predict[i] for i in range(3)}
```

Finally, we’ll install Gradio with Pip and create our Interface.

```py
# Installing and importing Gradio
!pip install gradio
import gradio as gr
sepal_length = gr.inputs.Slider(minimum=0, maximum=10, default=5, label="sepal_length")
sepal_width = gr.inputs.Slider(minimum=0, maximum=10, default=5, label="sepal_width")
petal_length = gr.inputs.Slider(minimum=0, maximum=10, default=5, label="petal_length")
petal_width = gr.inputs.Slider(minimum=0, maximum=10, default=5, label="petal_width")
gr.Interface(predict_flower, [sepal_length, sepal_width, petal_length, petal_width], "label", live=True, interpretation="default").launch(debug=True)
```

Output description

```
The legend tells you how changing that feature will affect the output. So increasing petal length and petal width will increase the confidence in the virginica class. Petal length is more “important” only in the sense that increasing petal length gets you “redder” (more confident) faster.
```

> [Terence Shin](https://www.linkedin.com/in/terenceshin/) [Understanding Feature Importance and How to Implement it in Python](https://towardsdatascience.com/understanding-feature-importance-and-how-to-implement-it-in-python-ff0287b20285#:~:text=Feature%20Importance%20refers%20to%20techniques,to%20predict%20a%20certain%20variable.)


[Feature Importance Correlation Coefficient]()
=

The correlation coefficient is a statistical measure of the strength of a linear relationship between two variables. Its values can range from -1 to 1. A correlation coefficient of -1 describes a perfect negative, or inverse, correlation, with values in one series rising as those in the other decline, and vice versa. A coefficient of 1 shows a perfect positive correlation, or a direct relationship. A correlation coefficient of 0 means there is no linear relationship.
<br>
Correlation coefficients are used in science and in finance to assess the degree of association between two variables, factors, or data sets. For example, since high oil prices are favorable for crude producers, one might assume the correlation between oil prices and forward returns on oil stocks is strongly positive. Calculating the correlation coefficient for these variables based on market data reveals a moderate and inconsistent correlation over lengthy periods.
<br>

Understanding the Correlation Coefficient
-

Different types of correlation coefficients are used to assess correlation based on the properties of the compared data. By far the most common is the Pearson coefficient, or Pearson's r, which measures the strength and direction of a linear relationship between two variables. The Pearson coefficient cannot assess nonlinear associations between variables and cannot differentiate between dependent and independent variables

The Pearson coefficient uses a mathematical statistics formula to measure how closely the data points combining the two variables (with the values of one data series plotted on the x-axis and the corresponding values of the other series on the y-axis) approximate the line of best fit. The line of best fit can be determined through regression analysis.

### Equations


- Correlation Coefficient

$$
\rho_{xy} = \frac{{Cov}(x,y)}{{\sigma}_{x}{\sigma}_{y}}
$$

- $ \rho_{xy}$ - Pearson product-moment correlation coefficient
- $ {Cov}(x,y) $ - covariance of variables x and y
- $ {\sigma}_{x} $ - standard deviation of x
- $ {\sigma}_{y} $ - standard deviation of y


Standard deviation is a measure of the dispersion of data from its average. Covariance shows whether the two variables tend to move in the same direction, while the correlation coefficient measures the strength of that relationship on a normalized scale, from -1 to 1.
<br>

The formula above can be elaborated as
<br>
$
r = \frac{n \ * \ (\sum{(X,Y)}) \ - \ (\sum(X) \ * \ \sum(Y))}{\sqrt{(n \ \sum({X^2}) \ - \ \sum(X)^2) \ * \ (n \ * \ \sum(Y^2) - \sum(Y)^2)}}
$

where:
- r=Correlation coefficient
- n=Number of observations
​

# TO:DO<!-- Add mutual information notes to note book -->
5. risk
6. mutual-info
7. correlation
8. one hot encoding
9. logistic regression
10. training log reg
11. log reg interpretation
12. using log regresion
13. summary
14. more exploration