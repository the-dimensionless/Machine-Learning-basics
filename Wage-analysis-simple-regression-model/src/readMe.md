# Notes and points to ponder


## Univariate Measures

### Mean

* Considers all values
* Mean is sensitive to extreme values (outliers greatly affect it)

### Median
> Separation of upper halves from lower ones.

* Insensitive to extreme values (closer to middle )
* Does not consider dataset distribution (cares only > or < a particular value)

### Percentile (% below a value)

* More expressive (realistic)
* Multiple measures

---------------------------------------------------
> All above methods are centered around middle data
---------------------------------------------------

### Standard Deviation
> Avg diff between data values and mean

* Considers all items
* Considers data distribution
* Harder to calculate

### Max, Min, Count, Mode, Range, Outliers (<> mean +- 2*SD)

## Bivariate measures

### Correlation

> Extent of linear relationships (+ve, -ve, no correlation)

> Correlation Fallacy : It does not imply in causality.

## Data Visuals (detect impossible values, identify data shape, detect errors in data)

### Distributions : 
    1. Normal
    2. Skewed
    3. Exponential

### Histograms
### Density (Distribution) Graphs
### Box and Whisker Plots
### Scatter Plots

## Data Scaling

* Standardization : Removing the mean and scaling to unit variance.
* MinMax Scaling : Rescale all attributes to range between zero and one.
* Normalization Scaling : Rescaling each observation to unit value.

## Data Segregation

## K Fold Cross Validation (Recommended k = 10) (accurate but k times slower)
* vs
## Pareto Principle for training/testing [80:20]
> Randomize the dataset

## Regression
> Bias/Variance tradeoff
* Linear Regression
* Ridge
* Lasso
* Elastic Net Regression
* Decision Tree Regressor
* SVM Regressor
* K-neighbours Regression

## Evaluation Metrics

* Max Error
* Mean Absolute Error
* R Squared

## Improve Models (--todo--)

* Ensemble methods
* Bagging
* Boosting
* Voting

## Deployment of Model
* Serialization and Deserialization (Pickle, Joblib)
* Web Services (REST)

## Logging
## Auto-healing
