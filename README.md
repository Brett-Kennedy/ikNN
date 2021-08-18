# ikNN

An sklearn-compatable predictor that aggregates predictions from the set of all 2d subspaces, using a standard kNN algoritm within each 2d space. This allows typically similar accuracy as standard kNNs (in some cases higher and others lower, but typically similar), such as sklearn's, but improved interpretability, as the 2d spaces may easily be visualized. Further, it allows reducing the influence of less-predictive features. 

This predictor follows the standard sklearn fit-predict model. 

The tool also provides visualizations, allowing this to function as a highly interpretable model. 

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


## Example Notebooks
One example notebook is provided, [Simple_Example_ikNN](https://github.com/Brett-Kennedy/ikNN/blob/main/examples/Simple_Example_ikNN.ipynb). This provides basic examples using the tool.

## Example Python Files
One file is provides to evalutat the overall accuracy of the model. [Accuracy_Test-ikNN](https://github.com/Brett-Kennedy/ikNN/blob/main/examples/Accuracy_Test_ikNN.py). This uses the [DatasetsEvaluator](https://github.com/Brett-Kennedy/DatasetsEvaluator) tool to compare the performance of ikNNs to standard sklearn kNNs. This measures accuracy only, as the interpretability can not easily be compared, but we believe it is safe to say that visualized 2d spaces are far more interpretable than high-dimensional spaces. 

## Results

The results of Accuracy_Test-ikNN.py are provided in the Results folder for one execution, using 100 random datasets for classification. 


| Model| 	Avg f1_macro	| Avg. Train-Test Gap	| Avg. Fit Time | 
| ----- |	----- |	----- |	----- |	
| kNN	| 0.659	| 0.107	| 0.003 | 
| ikNN	| 0.725	| 0.082	| 2.202 | 

![Line Graph](https://github.com/Brett-Kennedy/ikNN/blob/main/Results/results_17_08_2021_17_19_39_plot.png) This plots the accuracy of ikNN against two kNNs, with n_neighbors set to 5 and to 10.




