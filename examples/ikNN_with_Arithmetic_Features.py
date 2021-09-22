import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Todo: remove once have pip install
import sys
sys.path.insert(0, 'C:\python_projects\DatasetsEvaluator_project\DatasetsEvaluator')
import DatasetsEvaluator as de

sys.path.insert(0, 'C:\\python_projects\\ikNN_project\\ikNN')
from interpretable_knn import ikNNClassifier

sys.path.insert(0, 'C:\python_projects\ArithmeticFeatures_project\ArithmeticFeatures')
from ArithmeticFeatures import ArithmeticFeatures # todo: fix once have pip install

NUM_DATASETS = 100


def load_datasets(cache_folder):
    datasets_tester = de.DatasetsTester(
        problem_type="classification",
        path_local_cache=cache_folder)

    matching_datasets = datasets_tester.find_datasets(
        min_num_classes=2,
        max_num_classes=20,
        min_num_minority_class=5,
        max_num_minority_class=np.inf,
        min_num_features=0,
        max_num_features=np.inf,
        min_num_instances=500,
        max_num_instances=5_000,
        min_num_numeric_features=2,
        max_num_numeric_features=50,
        min_num_categorical_features=0,
        max_num_categorical_features=50)

    print("Number matching datasets found:", len(matching_datasets))

    # Note: some datasets may have errors loading or testing.
    datasets_tester.collect_data(
        max_num_datasets_used=NUM_DATASETS,
        use_automatic_exclude_list=True,
        method_pick_sets='pick_first',
        save_local_cache=True,
        check_local_cache=True,
        preview_data=False)

    return datasets_tester


def compare_accuracy_default(datasets_tester, partial_result_folder, results_folder):
    pipe1 = Pipeline([('model', KNeighborsClassifier())])
    pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', KNeighborsClassifier())])
    pipe3 = Pipeline([('model', ikNNClassifier())])
    pipe4 = Pipeline([('arith', ArithmeticFeatures()), ('model', ikNNClassifier())])

    summary_df, saved_file_name = datasets_tester.run_tests(
        estimators_arr=[
            ("kNN", "Original Features", "", pipe1),
            ("kNN", "Arithmetic-based Features", "", pipe2),
            ("ikNN", "Original Features", "", pipe3),
            ("ikNN", "Arithmetic-based Features", "", pipe4)
        ],
        partial_result_folder=partial_result_folder,
        results_folder=results_folder,
        show_warnings=False,
        run_parallel=True)

    datasets_tester.summarize_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)
    datasets_tester.plot_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)


def main():
    cache_folder = "c:\\dataset_cache"
    partial_result_folder = "c:\\intermediate_results"
    results_folder = "c:\\results"

    datasets_tester = load_datasets(cache_folder)
    compare_accuracy_default(datasets_tester, partial_result_folder, results_folder)


if __name__ == "__main__":
    main()
