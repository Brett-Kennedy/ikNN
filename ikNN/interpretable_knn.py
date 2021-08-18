import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.sparse import issparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from math import ceil, sqrt
from statistics import stdev, mode, mean

class ikNNClassifier(BaseEstimator, ClassifierMixin):
    """
    An sklearn-compatible predictor that aggregates predictions from the set of 2d spaces, using a standard
    kNN algorithm within each 2d space. This allows typically similar accuracy as standard kNNs, such as
    sklearn's (sometimes higher and sometimes lower), but improved interpretability, as the 2d spaces may
    easily be visualized. Further, it allows reducing the influence of less-predictive features. 

    This predictor follows the standard sklearn fit-predict model. 
    """

    def __init__(self, n_neighbors=8, method='use proba', weight_by_score=True, max_spaces_tested=50, num_best_spaces=5):
        """
        n_neighbors: int

        method: str
            must be 'simple majority' or 'use proba'. How the predictions are determined is based on this and 
            weight_by_score. If the method is 'simple majority', each kNN used in the predictions simply predicts the 
            majority within n_neighbors. If the method is 'use proba', each kNNs prediction is a distribution of
            predictions, weighted by their purity in the range of n_neighbors.  

        weight_by_score: bool
            If True, the kNNs used for each prediction will be weighted by their accuracy on the training set. 

        max_spaces_tested: int
            This may be set to control the time required for fitting where the dataset has a large number of columns. If 
            the number of pairs of numeric columns is greater than this, a subset will be used based on their accuracy
            as 1d kNN models. 

        num_best_spaces: int
            The number of 2d kNN's used for predictions. If -1, a kNN will be created for every pair of columns.
        """

        self.n_neighbors = n_neighbors
        self.method = method
        self.weight_by_score = weight_by_score
        self.max_spaces_tested = max_spaces_tested
        self.num_best_spaces = num_best_spaces

        self.X = []
        self.y = []
        self.num_columns = 0
        self.knn_arr = []
        self.cv_f1_score_arr = []
        self.col_pair_arr = []
        self.top_knns = []
        self.top_knns_visualize = []

    def fit(self, X, y):
        """
        Fit the k-nearest neighbors classifier from the training dataset. This creates a set of sklearn
        KNeighborsClassifier classifiers that are later ensembled to make predictions and visualized to present
        explanations.

        X: 2d array-like of shape (n_samples, n_features)

        y: array-like of shape (n_samples)
        """
        
        assert self.n_neighbors > 0
        assert self.method in ['simple majority', 'use proba']
        assert self.weight_by_score in [True, False]
        assert self.num_best_spaces==-1 or self.num_best_spaces>0

        if np.iscomplex(X).any() or np.iscomplex(y).any(): 
            raise ValueError("Complex data not supported")
        if (X.ndim!=2 or X.shape[0]==0 or X.shape[1]==0): 
            raise ValueError("Found array with 0 feature(s) (shape=(" + str(X.shape) + ") while a minimum of 1 is required.")        
        if issparse(X) or issparse(y):
            raise ValueError("Does not support sparse")
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.num_columns = self.X.shape[1]
        self.knn_arr = []
        self.cv_f1_score_arr = []
        self.col_pair_arr = []
        self.top_knns = []
        self.top_knns_visualize = [] 
        
        # If the number of pairs of columns is greater than the specified max_spaces_tested, we consider only a 
        # subset of the pairs of columns. Determine the best 1d kNNs and test max_spaces_tested pairs of the top 1d spaces. 
        cols_considered = list(range(self.num_columns))
        if (self.num_columns * (self.num_columns-1)/2) > self.max_spaces_tested:
            one_d_scores_arr = []
            for i in range(self.num_columns):
                col1_data = self.X[:,i].reshape(-1, 1)
                # KNeighborsClassifier requires a 2d input, so a second column of a constant value is used as well.
                col2_data = np.ones(len(X)).reshape(-1,1)
                two_d_set = np.hstack((col1_data,col2_data)) 
                clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                scores = cross_validate(clf, two_d_set, y, cv=5, scoring='f1_macro', return_train_score=False)
                test_scores = scores['test_score']
                one_d_scores_arr.append(mean(test_scores))
            top_one_d_scores_arr = np.argsort(one_d_scores_arr)[::-1]
            num_cols_used = ceil(sqrt(self.max_spaces_tested))
            cols_considered = top_one_d_scores_arr[:num_cols_used]

        # Create a list of sklearn knn's: one for each pair of features
        for i_idx in range(len(cols_considered)-1):
            i = cols_considered[i_idx]
            col1_data = self.X[:, i].reshape(-1, 1)
            for j_idx in range(i_idx+1, len(cols_considered)):
                j = cols_considered[j_idx]
                col2_data = self.X[:,j].reshape(-1, 1)
                two_d_set = np.hstack((col1_data, col2_data))
                
                # Assess the accuracy of each 2d kNN
                clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                scores = cross_validate(clf, two_d_set, y, cv=5, scoring='f1_macro', return_train_score=False)
                test_scores = scores['test_score']

                clf_final = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                clf_final.fit(two_d_set, y)
                self.knn_arr.append(clf_final)
                self.col_pair_arr.append((i,j))
                self.cv_f1_score_arr.append(test_scores.mean())

        # This sorts the kNNs by f1 score, so the visualizations can focus on the most accurate.                    
        self.top_knns_visualize = np.argsort(self.cv_f1_score_arr)[::-1]
        
        if self.num_best_spaces > 0:
            #self.top_knns = np.argsort(self.cv_f1_score_arr)[-self.num_best_spaces:]
            self.top_knns = self.top_knns_visualize[:self.num_best_spaces]
                
        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        d = {
            'n_neighbors': self.n_neighbors,
            'method': self.method,
            'weight_by_score': self.weight_by_score,
            'num_best_spaces': self.num_best_spaces
        }
        return d

    # Create a 2d numpy array with the 2 specified columns (indexes i and j) from X. X must be a numpy array
    def get_two_columns(self, X, i, j):
        col1_data = X[:,i].reshape(-1,1)
        col2_data = X[:,j].reshape(-1,1)
        two_d_set = np.hstack((col1_data, col2_data))
        return two_d_set
    
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        """
        X = np.array(X)
        y_classes = self.knn_arr[0].classes_
        
        # We track statistics about the predictions in order to visualize them later. Each array contains 
        # a row for each row in X. 
        # predict_arr: has the final prediction per row
        # subset_pred_arr_arr: has an array with final prediction of each 2d kNN for each row.
        # subset_pred_proba_arr_arr: has an element for each 2d kNN for each class for each row. 
        self.predict_arr = []
        self.subset_pred_arr_arr = []
        self.subset_pred_proba_arr_arr = [] 

        if (self.num_best_spaces > 0):
            clf_idxs = self.top_knns
        else:
            clf_idxs = range(len(self.knn_arr))
        
        for clf_idx in clf_idxs:
            clf = self.knn_arr[clf_idx]
            col_pair = self.col_pair_arr[clf_idx]
            two_d_set = self.get_two_columns(X, *col_pair)
            subset_pred_arr = clf.predict(two_d_set).reshape(-1,1)
            subset_pred_proba_arr = clf.predict_proba(two_d_set)
            if (len(self.subset_pred_arr_arr) == 0):
                self.subset_pred_arr_arr = subset_pred_arr
                self.subset_pred_proba_arr_arr = subset_pred_proba_arr
            else:
                self.subset_pred_arr_arr = np.hstack((self.subset_pred_arr_arr, subset_pred_arr))
                self.subset_pred_proba_arr_arr = np.hstack((self.subset_pred_proba_arr_arr, subset_pred_proba_arr))              

        for row_idx in range(len(X)):
            # Get the predictions and proba's for each 2d kNN used for this row. These are arrays with an element
            # for each 2d kNN. In the case the proba_collection, that element is an array, with an element for each 
            # target class.
            prediction_collection = self.subset_pred_arr_arr[row_idx, :]
            proba_collection = self.subset_pred_proba_arr_arr[row_idx, :]
            
            # The simple majority vote among the 2d kNN's. 
            majority = mode(prediction_collection)

            # Calculate the total (equivalent to average) probability per class among the 2d kNNs and take the 
            # class with the highest total probability. 
            proba_sum_per_class = [0]*len(y_classes)
            for i in range(0, len(proba_collection), len(y_classes)):
                for j in range(len(y_classes)):
                    proba_sum_per_class[j] += proba_collection[i+j]
            highest_avg = y_classes[proba_sum_per_class.index(max(proba_sum_per_class))]

            # Calculate the simple majority weighted by the f1 scores of the 2d kNNs on the training set. 
            weighted_sum_per_class = [0]*len(y_classes)
            for i, knn_idx in enumerate(clf_idxs):
                sub_prediction = prediction_collection[i] # the prediction of one 2d kNN
                sub_prediction_idx = np.where(y_classes==sub_prediction)[0][0]
                weighted_sum_per_class[sub_prediction_idx] += self.cv_f1_score_arr[knn_idx]
            highest_weighted_majority = y_classes[weighted_sum_per_class.index(max(weighted_sum_per_class))]

            # Calculate the average probably per class among the 2d kNNs and take the class with the 
            # highest total probability.
            weighted_proba_sum_per_class = [0]*len(y_classes)
            for class_idx in range(len(y_classes)):
                weighted_proba_sum_per_class[class_idx] = proba_sum_per_class[class_idx] * weighted_sum_per_class[class_idx]
            highest_weighted_avg = y_classes[weighted_proba_sum_per_class.index(max(weighted_proba_sum_per_class))]

            if (self.method=="simple majority"):
                if (self.weight_by_score == False):
                    self.predict_arr.append(majority)
                else:
                    self.predict_arr.append(highest_weighted_majority)
            else:
                if (self.weight_by_score == False):
                    self.predict_arr.append(highest_avg)
                else:
                    self.predict_arr.append(highest_weighted_avg)                
                
        return self.predict_arr

    def predict_proba(self, X):
        """ Return probability estimates for the test data X. """
        pass

    def score(self, X, y, sample_weight=None):
        """ Return the macro f1-score. """
        self.predict(X)
        return f1_score(self.predict_arr, y, average='macro')

    def set_params(self, **params):        
        """ Set the parameters of this estimator. """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    # row should be a pandas series
    def graph_2d_spaces(self, row, row_idx, true_class, num_spaces_shown=-1):
        """
        Presents a visualization for the fit model. The visualization consists of a set of plots for the row
        specified, where each plot represents one 2d space. The number of 2d spaces may be specified. The 2d spaces
        that are the most predictive, as measured on the train set, will be shown.

        Parameters
        ----------
        row: pandas series
            The row for which we want to show the predictions. The row will be shown as a red star in each 2d space.

        row_idx: int
            The index in the test set of row.

        true_class: int
            The index of the true class in y

        num_spaces_shown: int
            The number of 2d spaces plotted

        Returns
        -------
        None
        """

        tableau_palette_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink",
                                "tab:gray", "tab:olive", "tab:cyan"]
        
        if num_spaces_shown == -1:
            num_spaces_shown = min(5, len(self.col_pair_arr))
        else:
            num_spaces_shown = min(num_spaces_shown, len(self.col_pair_arr))
        if (self.num_best_spaces > 0):
            num_spaces_shown = min(num_spaces_shown, self.num_best_spaces)

        fig, ax = plt.subplots(nrows=1, ncols=num_spaces_shown, figsize=(25, 5))
        y_values = self.knn_arr[0].classes_
        x_df = pd.DataFrame(self.X)
        y_series = pd.Series(self.y)
        pred_by_space = self.subset_pred_arr_arr[row_idx]
        
        for plot_idx in range(num_spaces_shown):
            knn_idx = self.top_knns_visualize[plot_idx]
            clf = self.knn_arr[knn_idx]
            col_pair = self.col_pair_arr[knn_idx]
                        
            for class_idx in range(len(y_values)):
                class_name = y_values[class_idx]
                idx_arr = y_series.loc[y_series == class_name].index
                X_curr_class = x_df.loc[idx_arr] 
                
                ax[plot_idx].scatter(X_curr_class[X_curr_class.columns[col_pair[0]]], 
                                     X_curr_class[X_curr_class.columns[col_pair[1]]], 
                                     alpha=0.4, 
                                     c=tableau_palette_list[class_idx], 
                                     label=class_name)  
                
            ax[plot_idx].plot(row[col_pair[0]],
                              row[col_pair[1]],
                              marker="*",
                              markersize=15,
                              markeredgecolor="red",
                              markerfacecolor="red")
                
            x_min = ax[plot_idx].get_xlim()[0]
            x_max = ax[plot_idx].get_xlim()[1]
            x_step = (x_max-x_min)/100
            y_min = ax[plot_idx].get_ylim()[0]
            y_max = ax[plot_idx].get_ylim()[1]
            y_step = (y_max-y_min)/100   
            x_mesh, y_mesh = np.meshgrid((np.arange(x_min, x_max, x_step)), (np.arange(y_min, y_max, y_step)))
            df = pd.DataFrame({"0": x_mesh.reshape(-1), "1": y_mesh.reshape(-1)})
            mesh_pred = clf.predict(df)
            for i, uv in enumerate(y_values):
                mesh_pred[mesh_pred==uv] = i
            ax[plot_idx].contourf(x_mesh, y_mesh, mesh_pred.reshape(x_mesh.shape), alpha=0.1)                
                
            ax[plot_idx].legend()
            ax[plot_idx].set_title("Columns " + str(col_pair[0]) + ", " + str(col_pair[1]) + " -- Predicted: " + str(pred_by_space[plot_idx]))

        fig_title = "Row: " + str(row_idx) + " -- True Class: " + str(true_class) + " -- Predicted Class: " + str(self.predict_arr[row_idx])
        fig_title += " (Correct)" if self.predict_arr[row_idx] == true_class else " (Wrong)"
        fig.suptitle(fig_title)
        plt.show()
        