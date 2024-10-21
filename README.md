# ikNN

A scikit-learn-compatible predictor that aggregates predictions from the set of all 2d numeric subspaces within a dataset, using a standard kNN algoritm within each 2d space.  The model is named *ikNN* as it is an interpreretable form of kNN. The interpretability derives from the fact that the set of 2d spaces are each visualizable, providing a full explanation based on the training data. 

As may be expected, as this is an ensembling approach, the accuracy is, from our testing to date, higher than that of standard kNN models. But, despite being an ensembling approach, which typically preclude interpretability, this allows full global and local explanations, that is, descriptions of the overall model as well as explanations of individual predictions.  

This predictor follows the standard sklearn fit-predict model. Currently only classification is available, with regression in future versions. 

ikNN provides, in effect, an ensembling method specific to kNNs, though the general techique is based on weighted voting. Although a straightforward design, testing suggests it can be a quite accurate and interpretable model. 

For a quick summary, see: https://towardsdatascience.com/interpretable-knn-iknn-33d38402b8fc 

## Algorithm
The model first examines each pair of features and creates a standard 2d kNN using these features and assesses their accuracy with respect to predicting the target column using the training data. Given this, the ikNN model determines the predictive power of each 2d subspace. To make a prediction, the 2d subspaces known to be most predictive are used, optionally weighted by their predictive power on the training data. Further, at inference, the purity of the set of neareast neighbors around a given row within each 2d space may be considered, allowing the model to weight more heavily both the subspaces proven to be more predictive with training data and the subspaces that appear to be the most uniform in their prediction with respect to the current instance. 

This approach allows the model to consider the influence all input features, but weigh them in a manner that magnifies the influence of more predictive features, and diminishes the influence of less-predictive features. 

As with standard kNN's, any categorical columns must be numerically encoded. 

## Installation

The tool requires only the interpretable_knn.py file. This may be downloaded and included in any project.


## Examples

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate

iris = load_iris()
X, y = iris.data, iris.target

clf = ikNNClassifier()
scores = cross_validate(clf, X, y, cv=5, scoring='f1_macro', return_train_score=False)
```

This example provides an example using cross validation. The standard fit-predict methodology may also be used:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```


## Example Notebooks
One example notebook is provided, [Simple_Example_ikNN](https://github.com/Brett-Kennedy/ikNN/blob/main/examples/Simple_Example_ikNN.ipynb). This provides basic examples using the tool.

The tool provides the option to visualize an explanation for each prediction, as in the following example found in the notebook:

![Line Graph](https://github.com/Brett-Kennedy/ikNN/blob/main/Results/ikNN_Output.png)

## Visualizations
The number of 2d spaces provided for each row explanation is configurable, but in most cases only a small number of plots need be and are shown. With datasets with a large number of columns, showing each pair is unmanageable and not useful. Although it is configurable, by default, only five 2d spaces are used by each ikNN. This ensures the prediction times are fast and the visualizations simple, as only this many 2d spaces need be shown. It should be noted, that for most datasets, for most rows, all or almost all 2d spaces agree on the prediction. However, where the predictions are incorrect, it may be useful to examine more 2d plots in order to better tune the hyperparameters to suit the current dataset. 

## Example Python Files
Two files are provided to evaluate the overall accuracy of the model. The first is [Accuracy_Test-ikNN](https://github.com/Brett-Kennedy/ikNN/blob/main/examples/Accuracy_Test_ikNN.py). This uses the [DatasetsEvaluator](https://github.com/Brett-Kennedy/DatasetsEvaluator) tool to compare the performance of ikNNs to standard sklearn kNNs. This measures accuracy only, as the interpretability can not easily be compared, but we believe it is safe to say that visualized 2d spaces are far more interpretable than high-dimensional spaces. This creates 2 sets of output files: 1 given [default parameters](https://github.com/Brett-Kennedy/ikNN/tree/main/Results/Default%20Parameters) and one using a [grid search to determine the best hyperparameters](https://github.com/Brett-Kennedy/ikNN/tree/main/Results/Grid%20Search%20Best%20Parameters)

Using DatasetsEvaluator provides an unbiased (as the datasets selected are selected randomly) and straightforward method to test on a large number of datasets in a repeatable manner. 

The second, is [ikNN_with_Arithmetic_Features](https://github.com/Brett-Kennedy/ikNN/blob/main/examples/ikNN_with_Arithmetic_Features.py). This is similar, but tests ikNN in conjunction with [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), a feature engineering tool available on this github site. The [results](https://github.com/Brett-Kennedy/ikNN/tree/main/Results/With%20Arithimetic%20Features) may be viewed directly; they are also summarized below.  

As ArithmeticFeatures produces interpretable features (simple arithetic operations on pairs of numeric features), the ikNN models produced using these features are still highly, though slightly less, interpretable than ikNN models using only the original features. 

## Results

### Default Parameters

The results of Accuracy_Test-ikNN.py are provided in the Results folder for one execution, using 100 random datasets for classification. Using DatasetsEvaluator, we get 4 output files for each test:
1. A csv file listing every test (every model type on every dataset).
2. A csv file summarizing this. This has one row for each model type and aggregate statistics over all datasets
3. A line plot listing the performance of each model on each dataset. The datasets are listed along the x-axis, with an accuracy score (in this case macro F1 score) on the y-axis. Each line represents one model type. The datasets are ordered from left to right based on the accuracy given the model specified as the baseline against which we compare, in this case a standard kNN with k=5 and all other parameters set to their default value. 
4. a heatmap indicating how often each model was the top performer in terms of accuracy and interpretability. In the case of kNN and ikNN models, there is no objective measure of interpretability, so this plot simply records how often each model performed the best. 

| Model| 	Avg f1_macro	| Avg. Train-Test Gap	| Avg. Fit Time | 
| ----- |	----- |	----- |	----- |	
| kNN	| 0.659	| 0.107	| 0.003 | 
| ikNN	| 0.725	| 0.082	| 2.202 | 

|Model | # Times the Highest Accuracy | 
| ----- |	----- |
| kNN (k=5) | 19 |
| kNN (k=10)| 23 | 
| ikNN | 58 | 

![Line Graph](https://github.com/Brett-Kennedy/ikNN/blob/main/Results/Default%20Parameters/results_22_09_2021_13_24_04_plot.png) This plots the accuracy of ikNN against two kNNs, with n_neighbors set to 5 and to 10.

Here, the green line represents the accuracy of the ikNN, while the orange and blue lines represent the accuracy of standard kNN's, with k=5 and 10. Otherwise, default parameters are used in all 3 models. The accuracy of the ikNN can be seen to be very competitive with standard kNNs, and typically significantly higher.

It should be noted, ikNN's are somewhat slower than standard kNNs, though still fitting and predicting within seconds, and as such, still compare favourably to many other models, particularly other ensembled models and neural networks with regards to time. 

Note as well, the gap between the train and test scores is, on average, smaller for ikNN than kNN models, suggesting ikNN's are more stable, as is expected from an ensemble model.

### Using CV Grid Search to Identify the Best Hyperparameters

| Model| 	Avg f1_macro	| Avg. Train-Test Gap	|
| ----- |	----- |	----- |	
| kNN	| 0.685	| 0.076	| 
| ikNN	| 0.775	|0.083 |

|Model | # Times the Highest Accuracy | 
| ----- |	----- |
| kNN | 24 |
| ikNN | 76 | 

![Line Graph](https://github.com/Brett-Kennedy/ikNN/blob/main/Results/Grid%20Search%20Best%20Parameters/results_21_09_2021_18_52_32_plot.png)
Here, again, we see ikNN significantly outperforming standard kNNs. Note, however, with kNNs, this experiment tuned only the value of k, while with ikNN models, we tuned 3 additional hyperparameters: method, weight_by_score, and max_spaces_tested. It should be noted, all four hyperparameters tuned for ikNN's frequently selected all values made possible. While ikNNs have sensible values for these set by default, tuning may be beneficial. It also appears that ikNNs are less sensitive to the value of k selected, though analysis of this is ongoing.



### Tesing with ArithmeticFeatures

| Model| 	Features | Avg f1_macro	| Avg. Train-Test Gap	| Avg. Fit Time | 
| ----- |	----- |	----- | ----- |	----- |	
|kNN	|Original Features	|0.660	|0.126 |	0.006 |
|kNN	|Arithmetic-based Features	|0.634 |	0.134 |	13.813|
|ikNN	|Original Features	|0.721 |	0.0798 |	3.802 |
|ikNN	|Arithmetic-based Features	|0.745|	0.083|	80.489|


|Model | Features | # Times the Highest Accuracy | 
| ----- |	----- |	----- |
|kNN | Original  | 20 |
|kNN | Original + Arithmetic  | 13 |
|ikNN | Original  | 12 |
|ikNN | Original + Arithmetic  | 55 |

![Line Graph](https://github.com/Brett-Kennedy/ikNN/blob/main/Results/With%20Arithimetic%20Features/results_21_09_2021_16_42_21_plot.png)

Here, the blue and orange lines represent kNNs and green and red ikNNs. Interestingly, though the use of ArithmeticFeatures tended to lower the accuracy of standard kNN models, it tended to raise the accuracy of ikNN models, making them even stronger models. With ikNNs, we see the opposite behaviour, with the red line tending to follow the shape of the greeen, but slightly higher, indicating higher accuracy.

While the use of ArithmeticFeatures can lead to overfitting in some contexts, as it generates a large number of features, ikNN is able to take advantage of these features well. In general, ikNN is able to absorb engineered features well, an artifact of its use of simple, 2d spaces, where one or both features may be engineered. 

## Methods

### ikNNClassifier()

```
iknn = ikNNClassifier(n_neighbors=15, 
                      method='simple majority', 
                      weight_by_score=True, 
                      max_spaces_tested=500, 
                      num_best_spaces=6)
```
#### Parameters

**n_neighbors**: int

The number of neighbors each rows is compared to. 

**method**: str
            
Must be 'simple majority' or 'use proba'. How the predictions are determined is based on this and 
weight_by_score. If the method is 'simple majority', each kNN used in the predictions simply predicts the 
majority within n_neighbors. If the method is 'use proba', each kNNs prediction is a distribution of
            predictions, weighted by their purity in the range of n_neighbors.  

**weight_by_score**: bool
If True, the kNNs used for each prediction will be weighted by their accuracy on the training set. 

**max_spaces_tested**: int
This may be set to control the time required for fitting where the dataset has a large number of columns. If 
            the number of pairs of numeric columns is greater than this, a subset will be used based on their accuracy
            as 1d kNN models. 

**num_best_spaces**: int
The number of 2d kNN's used for predictions. If -1, a kNN will be created for every pair of columns.

##

### Fit
```
iknn.fit(X,y)
```

Fit the k-nearest neighbors classifier from the training dataset. This creates a set of sklearn
        KNeighborsClassifier classifiers that are later ensembled to make predictions and visualized to present
        explanations.

#### Parameters
**X**: 2d array-like of shape (n_samples, n_features)

**y**: array-like of shape (n_samples)
   
##

### Predict 

```
y_pred = iknn.predict(X)
```

Predict the class labels for the provided data.
   
##

### graph_2d_spaces

```
graph_2d_spaces(row, row_idx, true_class, num_spaces_shown=-1)   
```

Presents a visualization for the fit model. The visualization consists of a set of plots for the row
        specified, where each plot represents one 2d space. The number of 2d spaces may be specified. The 2d spaces
        that are the most predictive, as measured on the train set, will be shown.

#### Parameters
       
**row**: pandas series

The row for which we want to show the predictions. The row will be shown as a red star in each 2d space.

**row_idx**: int

The index in the test set of row.

**true_class**: int

The index of the true class in y

**num_spaces_shown**: int

The number of 2d spaces plotted


