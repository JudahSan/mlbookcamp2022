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

# TO:DO
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