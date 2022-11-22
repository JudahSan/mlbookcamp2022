BINARY CLASSIFICATION
=

CHURN PREDICTION
-

$ g(x_i) ≈ y_i $

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
- View dat
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
    