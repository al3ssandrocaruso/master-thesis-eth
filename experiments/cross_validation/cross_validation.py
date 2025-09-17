import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from itertools import product
from utils.clustering.clustering_utils import get_embeddings
from utils.questionnaires.bdi import expand_bdi_data
from config.config import PATH_bdi, PATH_DEMOGRAPHIC_local, PATH_DEMOGRAPHIC_cluster

def main(args):
    # Load and prepare data
    trained_embeddings_path = args.trained_embeddings_path
    embeddings, metadata = get_embeddings(trained_embeddings_path)
    metadata_df = pd.DataFrame(metadata, columns=['user', 'day', 'window'])
    embeddings_df = pd.DataFrame(embeddings)
    combined_df = pd.concat([metadata_df, embeddings_df], axis=1)
    combined_df['features'] = combined_df.iloc[:, 3:].values.tolist()
    final_df = combined_df[['user', 'day', 'window', 'features']]
    final_df = final_df[final_df['window'] == 1]

    # Load and prepare CSV
    df_bdi = pd.read_csv(PATH_bdi)
    df_bdi['bdi_time'] = pd.to_datetime(df_bdi['bdi_time'])
    df_bdi['day'] = df_bdi['bdi_time'].dt.date
    df_bdi = df_bdi.drop(columns=['bdi_time'])
    df_bdi.rename(columns={'study_id': 'user'}, inplace=True)
    expanded_bdi = expand_bdi_data(df_bdi)

    data = pd.merge(final_df, expanded_bdi, on=['user', 'day'], how='inner')

    mean_bdi_for_user = data.groupby('user')['bdi_score'].mean()
    mean_bdi_for_user_df = pd.DataFrame(mean_bdi_for_user)

    # Define the categories
    def categorize_bdi(score):
        if score < 10:
            return 0
        elif 10 <= score < 19:
            return 1
        elif 19 <= score < 29:
            return 2
        else:
            return 3

    # Apply the categorization
    mean_bdi_for_user_df['Category'] = mean_bdi_for_user_df['bdi_score'].apply(categorize_bdi)
    all_users = mean_bdi_for_user_df.index.tolist()
    y_users = mean_bdi_for_user_df['Category'].tolist()

    # Map categories to binary values for binary classification
    def map_to_binary(value):
        if value in [0, 1]:
            return 0
        else:
            return 1

    y_users_binary = [map_to_binary(y) for y in y_users]

    X = pd.DataFrame(data['features'].tolist())
    X.columns = [f'feat_{i + 1}' for i in range(X.shape[1])]
    y = data[[f'bdi_{i}' for i in range(1, 22)]]  # Select all BDI columns as labels

    df_participants = pd.read_csv(PATH_DEMOGRAPHIC_local)
    depressed_users = df_participants[df_participants['type'] == 'p']['user'].values

    # Define custom scoring functions
    def custom_accuracy_score(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracies = []
        for i in range(y_true.shape[1]):
            accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))
        return np.mean(accuracies)

    def custom_precision_score(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        precisions = []
        for i in range(y_true.shape[1]):
            precisions.append(precision_score(y_true[:, i], y_pred[:, i], average='macro', zero_division=1))
        return np.mean(precisions)

    def custom_recall_score(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        recalls = []
        for i in range(y_true.shape[1]):
            recalls.append(recall_score(y_true[:, i], y_pred[:, i], average='macro', zero_division=1))
        return np.mean(recalls)

    # Create scorers
    accuracy_scorer = make_scorer(custom_accuracy_score)
    precision_scorer = make_scorer(custom_precision_score)
    recall_scorer = make_scorer(custom_recall_score)

    # Define classifiers to be tested
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVC': SVC(random_state=42),
        'NaiveBayes': GaussianNB()
    }

    # Define the parameter grid for each classifier
    param_grid = {
        'LogisticRegression': {
            'C': [0.01, 0.1],
            'solver': ['lbfgs', 'saga'],
        },
        'KNN': {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance'],
        },
        'SVC': {
            'C': [0.001, 0.01],
            'kernel': ['linear', 'rbf'],
        },
        'DecisionTree': {
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
        },
        'NaiveBayes': {
            # No hyperparameters to tune for GaussianNB in this case
        }
    }
    # Flag to toggle user cross-validation
    user_cross_validation = True

    # Stratified K-Fold for cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    kf_sample = KFold(n_splits=10, shuffle=True, random_state=0)
    for clf_name, clf in classifiers.items():
        param_grid_clf = param_grid.get(clf_name, {})
        param_combinations = list(product(*param_grid_clf.values()))

        for params in param_combinations:
            param_dict = dict(zip(param_grid_clf.keys(), params))

            accuracies = []
            precisions = []
            recalls = []

            binary_accuracies = []
            binary_precisions = []
            binary_recalls = []

            if user_cross_validation:
                # User-based cross-validation
                for fold, (train_idx, test_idx) in enumerate(kf.split(all_users, y_users)):
                    train_users = [all_users[i] for i in train_idx]
                    test_users = [all_users[i] for i in test_idx]

                    X_train = X[data['user'].isin(train_users)]
                    y_train = y[data['user'].isin(train_users)]
                    X_test = X[data['user'].isin(test_users)]
                    y_test = y[data['user'].isin(test_users)]

                    clf.set_params(**param_dict)
                    pipeline = Pipeline([
                        ('classifier', MultiOutputClassifier(clf))
                    ])

                    pipeline.fit(X_train, y_train)

                    y_pred = pipeline.predict(X_test)

                    accuracy_per_bdi = custom_accuracy_score(y_test.values, y_pred)
                    precision_per_bdi = custom_precision_score(y_test.values, y_pred)
                    recall_per_bdi = custom_recall_score(y_test.values, y_pred)

                    accuracies.append(accuracy_per_bdi)
                    precisions.append(precision_per_bdi)
                    recalls.append(recall_per_bdi)

                    # Compute binary classification metrics
                    y_test_binary = y_test.applymap(map_to_binary)
                    y_pred_binary = pd.DataFrame(y_pred).applymap(map_to_binary)

                    binary_accuracy_per_bdi = custom_accuracy_score(y_test_binary.values, y_pred_binary.values)
                    binary_precision_per_bdi = custom_precision_score(y_test_binary.values, y_pred_binary.values)
                    binary_recall_per_bdi = custom_recall_score(y_test_binary.values, y_pred_binary.values)

                    binary_accuracies.append(binary_accuracy_per_bdi)
                    binary_precisions.append(binary_precision_per_bdi)
                    binary_recalls.append(binary_recall_per_bdi)
            else:
                # Sample-based cross-validation
                for fold, (train_idx, test_idx) in enumerate(kf_sample.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    clf.set_params(**param_dict)
                    pipeline = Pipeline([
                        ('classifier', MultiOutputClassifier(clf))
                    ])

                    pipeline.fit(X_train, y_train)

                    y_pred = pipeline.predict(X_test)

                    accuracy_per_bdi = custom_accuracy_score(y_test.values, y_pred)
                    precision_per_bdi = custom_precision_score(y_test.values, y_pred)
                    recall_per_bdi = custom_recall_score(y_test.values, y_pred)

                    accuracies.append(accuracy_per_bdi)
                    precisions.append(precision_per_bdi)
                    recalls.append(recall_per_bdi)

                    # Compute binary classification metrics
                    y_test_binary = y_test.applymap(map_to_binary)
                    y_pred_binary = pd.DataFrame(y_pred).applymap(map_to_binary)

                    binary_accuracy_per_bdi = custom_accuracy_score(y_test_binary.values, y_pred_binary.values)
                    binary_precision_per_bdi = custom_precision_score(y_test_binary.values, y_pred_binary.values)
                    binary_recall_per_bdi = custom_recall_score(y_test_binary.values, y_pred_binary.values)

                    binary_accuracies.append(binary_accuracy_per_bdi)
                    binary_precisions.append(binary_precision_per_bdi)
                    binary_recalls.append(binary_recall_per_bdi)

            # Calculate average metrics across all folds for this configuration
            mean_accuracy = np.mean(accuracies)
            mean_precision = np.mean(precisions)
            mean_recall = np.mean(recalls)

            mean_binary_accuracy = np.mean(binary_accuracies)
            mean_binary_precision = np.mean(binary_precisions)
            mean_binary_recall = np.mean(binary_recalls)

            # Print the summary for this specific model configuration
            print(f"Summary for {clf_name} with Params {param_dict}:")
            print(f"Multi-Class - Average Accuracy across folds: {mean_accuracy:.4f}")
            print(f"Multi-Class - Average Precision across folds: {mean_precision:.4f}")
            print(f"Multi-Class - Average Recall across folds: {mean_recall:.4f}")

            print(f"Binary Classification - Average Accuracy across folds: {mean_binary_accuracy:.4f}")
            print(f"Binary Classification - Average Precision across folds: {mean_binary_precision:.4f}")
            print(f"Binary Classification - Average Recall across folds: {mean_binary_recall:.4f}")
            print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation with cross-validation")
    parser.add_argument('--trained_embeddings_path', type=str, required=True, help='Path to the trained embeddings file')
    args = parser.parse_args()
    main(args)
