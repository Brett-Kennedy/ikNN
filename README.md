# ikNN

An sklearn-compatable predictor that aggregates predictions from the set of all 2d numeric subspaces within a dataset, using a standard kNN algoritm within each 2d space.  The model is named *ikNN* as it is an interpreretable form of kNN. The interpretability derives from the fact that the set of 2d spaces are each visualizable, providing a full explanation based on the training data. 

As may be expected, as this is an ensembling approach, the accuracy is, from our testing to date, higher than that of standard kNN models. But, despite being an ensembling approach, which typically preclude interpretability, this allows full global and local explanations, that is, descriptions of the overall model as well as explanations of individual predictions.  

This predictor follows the standard sklearn fit-predict model. Currently only classification is available, with regression in progress. 

## Algorithm
The model first examines each pair of features and creates a standard 2d kNN using these features and assesses their accuracy with respect to predicting the target column. Given this, the ikNN model determines the predictive power of each 2d subspace. To make a prediction, the 2d subspaces known to be most predictive are used, optionally weighted by their predictive power on the training data. Further, at inference, the purity of the set of neareast neighbors around a given row within each 2d space may be considered, allowing the model to weight more heavily both the subspaces proven to be more predictive with training data and the subspaces that appear to be the most uniform in their prediction with respect to the current row. 

This approach allows the model to consider the influence all input features, but weigh them in a manner that magnifies the influence of more predictive features, and diminishes the influence of less-predictive features. 

As with standard kNN's, any categorical columns must be numerically encoded. 

## Installation

`
pip install ikNN
`

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
One file is provided to evaluate the overall accuracy of the model. [Accuracy_Test-ikNN](https://github.com/Brett-Kennedy/ikNN/blob/main/examples/Accuracy_Test_ikNN.py). This uses the [DatasetsEvaluator](https://github.com/Brett-Kennedy/DatasetsEvaluator) tool to compare the performance of ikNNs to standard sklearn kNNs. This measures accuracy only, as the interpretability can not easily be compared, but we believe it is safe to say that visualized 2d spaces are far more interpretable than high-dimensional spaces. 

## Results

The results of Accuracy_Test-ikNN.py are provided in the Results folder for one execution, using 100 random datasets for classification. 


| Model| 	Avg f1_macro	| Avg. Train-Test Gap	| Avg. Fit Time | 
| ----- |	----- |	----- |	----- |	
| kNN	| 0.659	| 0.107	| 0.003 | 
| ikNN	| 0.725	| 0.082	| 2.202 | 

![Line Graph](https://github.com/Brett-Kennedy/ikNN/blob/main/Results/results_17_08_2021_17_19_39_plot.png) This plots the accuracy of ikNN against two kNNs, with n_neighbors set to 5 and to 10.

Here, the blue line represents the accuracy of the ikNN, while the orange and green lines represent the accuracy of standard kNN's, with k=5 and 10. Otherwise, default parameters are used in all 3 models. The accuracy of the ikNN can be seen to be very competitive with standard kNNs, and typically significantly higher.

It should be noted, ikNN's are somewhat slower than standard kNNs, though still fitting and predicting within seconds, and as such, still compare favourably to many other models, particularly other ensembled models and deep neural networks with regards to time. 

Note as well, the gap between the train and test scores is, on average, smaller for ikNN than kNN models, suggesting ikNN's are more stable, as is expected from an ensemble model.



