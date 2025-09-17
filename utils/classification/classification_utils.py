import numpy as np
from config.config import PATH_EMA, BASE_DIR
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import pickle
from sklearn.model_selection import KFold, GroupKFold
import pandas as pd
#import shap
#import matplotlib.pyplot as plt

# Define classifiers and parameter grids in a unified structure
classifiers_with_params = {
    'LogisticRegression': {
        'classifier': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.01, 0.1],
            'solver': ['lbfgs', 'saga'],
        }
    },
    'KNN': {
        'classifier': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance'],
        }
    },
    'SVC': {
        'classifier': SVC(random_state=42,probability=True),
        'params': {
            'C': [0.001, 0.01],
            'kernel': ['linear', 'rbf'],
        }
    },
    'DecisionTree': {
        'classifier': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
        }
    },
    'NaiveBayes': {
        'classifier': GaussianNB(),
        'params': {
            # No hyperparameters to tune for GaussianNB in this case
        }
    },
    'XGBoost': {
        'classifier': xgb.XGBClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
        }
    }
}


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

# Map categories to binary values for binary classification
def map_to_binary(value):
    if value in [0, 1]:
        return 0
    else:
        return 1

# Define custom scoring functions
def custom_accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracies = []
    for i in range(y_true.shape[1]):
        accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    return np.mean(accuracies)

# Custom scoring function using balanced accuracy
def balanced_accuracy(y_true, y_pred):
    # Using sklearn's balanced_accuracy_score for multi-class, multi-label classification
    return balanced_accuracy_score(y_true, y_pred, average='macro')  # 'macro' averages the recall of each label

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

def load_ema_quests(file):
    quest_data = pd.read_csv(file)
    # Remove the first column
    quest_data = quest_data.iloc[:, 1:]
    # Rename columns
    quest_data = quest_data.rename(columns={"Participant_ID": "user", "Date": "day"})
    
    return quest_data

def prepare_data(embeddings, metadata, expanded_bdi, df_participants, demo, mood):
    # Combine embeddings and metadata, and optionally merge with demographic and mood data
    metadata_df = pd.DataFrame(metadata, columns=['user', 'day', 'window'])
    embeddings_df = pd.DataFrame(embeddings)
    combined_df = pd.concat([metadata_df, embeddings_df], axis=1)

    if mood:
        df_quest = load_ema_quests(PATH_EMA)
        combined_df = pd.merge(combined_df, df_quest, on=['user', 'day'], how='inner')

    if demo:
        df_participants = df_participants[['user', 'Age', 'Sex']]
        df_participants['Sex'] = df_participants['Sex'].map({'male': 0, 'female': 1})
        combined_df = pd.merge(combined_df, df_participants, on='user', how='inner')

    combined_df = combined_df.dropna()
    combined_df['features'] = combined_df.iloc[:, 3:].values.tolist()

    final_df = combined_df[['user', 'day', 'window', 'features']]
    final_df = final_df[final_df['window'] == '1']

    data = pd.merge(final_df, expanded_bdi, on=['user', 'day'], how='inner')

    X = pd.DataFrame(data['features'].tolist())
    X.columns = [f'feat_{i + 1}' for i in range(X.shape[1])]
    y = data[[f'bdi_{i}' for i in range(1, 22)]]
    y = y.applymap(lambda x: 1 if x in [2, 3] else 0)

    return X, y, data

def perform_classification(clf_param_dict, embeddings_train, metadata_train, embeddings_val, metadata_val, embeddings_test, metadata_test, expanded_bdi, df_participants, demo=False, mood=False):

    # Prepare training data (train split)
    X_train, y_train, _ = prepare_data(
        embeddings_train, metadata_train, expanded_bdi, df_participants, demo, mood
    )
    
    # Prepare validation data (val split)
    X_val, y_val, _ = prepare_data(
        embeddings_val, metadata_val, expanded_bdi, df_participants, demo, mood
    )
    
    # Prepare test data (test split)
    X_test, y_test, _ = prepare_data(
        embeddings_test, metadata_test, expanded_bdi, df_participants, demo, mood
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.transform(X_test)

     # Hyperparameter grid search with classifier selection
    best_score = -float("inf")
    best_params = None
    best_clf = None

    for clf_name, clf_and_params in clf_param_dict.items():
        clf = clf_and_params['classifier']
        param_grid = clf_and_params['params']
        for params in ParameterGrid(param_grid):
            clf.set_params(**params)
            pipeline = Pipeline([
                ('classifier', MultiOutputClassifier(clf))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            val_score = f1_score(y_pred, y_val,average='macro')  # Or another metric

            if val_score > best_score:
                best_score = val_score
                best_params = params
                best_clf = clf_name

    # Train final model on combined train + val with the best classifier and hyperparameters
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.vstack((y_train, y_val))
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    final_clf = classifiers_with_params[best_clf]['classifier']
    final_clf.set_params(**best_params)
    pipeline = Pipeline([
        ('classifier', MultiOutputClassifier(final_clf))
    ])
    pipeline.fit(X_train_val, y_train_val)

    # Test set evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    results = {
        "best_clf": best_clf,
        "best_params": best_params,
        "y_test": y_test.values,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    return results

def get_fold_indices(data, groupkfold=True):
    """
    Generate fold indices for cross-validation with GroupKFold or standard KFold.

    Parameters:
        data (pd.DataFrame): Input data containing a 'user' column for grouping.
        groupkfold (bool): Whether to use GroupKFold (True) or standard KFold (False).

    Returns:
        List[Tuple[List[int], List[int], List[int]]]: A list of tuples, where each tuple contains
                                                      train indices, validation indices, and test indices.
    """
    total_size = len(data)
    indices = np.arange(total_size)

    # Extract the groups (user IDs)
    groups = data['user'].values

    if groupkfold:
        kf = GroupKFold(n_splits=10)
    else:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    folds = []

    # Create the splits with GroupKFold or KFold
    all_folds = list(kf.split(indices, groups=groups if groupkfold else None))

    for test_fold_idx in range(len(all_folds)):
        # Test fold
        _, test_indices = all_folds[test_fold_idx]
        
        # Validation fold (cyclically pick the next fold)
        val_fold_idx = (test_fold_idx + 1) % len(all_folds)
        _, val_indices = all_folds[val_fold_idx]

        # Training folds (all other folds except test and validation)
        train_indices = []
        for fold_idx, (_, train_fold) in enumerate(all_folds):
            if fold_idx != test_fold_idx and fold_idx != val_fold_idx:
                train_indices.extend(train_fold)

        folds.append((train_indices, val_indices, test_indices))

    return folds

def get_shap_explainer(individual_clf, X_test, background_data=None):
    """
    Returns the appropriate SHAP explainer for a given classifier.

    Args:
        individual_clf: Trained classifier for which SHAP explainer is required.
        X_test: Data to be explained.
        background_data: Reference data for KernelExplainer (if needed). Defaults to X_test.
    
    Returns:
        SHAP explainer object.
    """
    if isinstance(individual_clf, xgb.XGBClassifier):  # XGBoost uses TreeExplainer
        explainer = shap.TreeExplainer(individual_clf)
    elif isinstance(individual_clf, LogisticRegression):  # Use LinearExplainer for linear models
        explainer = shap.LinearExplainer(individual_clf, background_data or X_test)
    elif isinstance(individual_clf, DecisionTreeClassifier):  # Decision Trees use TreeExplainer
        explainer = shap.TreeExplainer(individual_clf)
    elif isinstance(individual_clf, SVC):  # SVC is not tree-based; KernelExplainer is suitable
        explainer = shap.KernelExplainer(individual_clf.predict, background_data or X_test)
    elif isinstance(individual_clf, (KNeighborsClassifier,GaussianNB)):  # k-NN is non-parametric; use KernelExplainer
        explainer = shap.KernelExplainer(individual_clf.predict, background_data or X_test)
    else:
        # Default to KernelExplainer for models not explicitly handled
        explainer = shap.KernelExplainer(individual_clf.predict, background_data or X_test)
    
    return explainer

def perform_classification_ema(clf_param_dict, expanded_bdi, df_participants):

    results = []

    all_X_test = pd.DataFrame()  # To store combined X_test data across all folds

    df_quest = load_ema_quests(PATH_EMA)
    if df_participants:
        df_participants = df_participants[['user', 'Age', 'Sex']]
        df_participants['Sex'] = df_participants['Sex'].map({'male': 0, 'female': 1})
        combined_df = pd.merge(df_quest, df_participants, on='user', how='inner')
    else:
        combined_df = df_quest

    combined_df = combined_df.dropna()
    feature_names = combined_df.iloc[:,2:].columns.values
    combined_df['features'] = combined_df.iloc[:, 2:].values.tolist()
    final_df = combined_df[['user', 'day', 'features']]
    bdi_columns = [f'bdi_{i}' for i in range(1, 22) if i != 9]

    data = pd.merge(final_df, expanded_bdi, on=['user', 'day'], how='inner')

    X = pd.DataFrame(data['features'].tolist())
    X.columns = [f'feat_{i + 1}' for i in range(X.shape[1])]
    y = data[bdi_columns]
    y = y.applymap(lambda x: 1 if x in [2, 3] else 0)

    folds = get_fold_indices(data, groupkfold=True)
    
    n_fold = 0

    # Store SHAP values for each output (symptom) across all folds
    shap_values_per_output_all_folds = {symptom: [] for symptom in y.columns}

    for fold in folds:
        print(f"Processing fold {n_fold}")

        train_idx, val_idx, test_idx = fold

        X_train = X.iloc[train_idx, :]
        X_val = X.iloc[val_idx, :]
        X_test = X.iloc[test_idx, :]

        y_train = y.iloc[train_idx, :]
        y_val = y.iloc[val_idx, :]
        y_test = y.iloc[test_idx, :]

        # Store \( X_{\text{test}} \) for this fold
        all_X_test = pd.concat([all_X_test, X_test], axis=0)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Hyperparameter grid search with classifier selection
        best_score = -float("inf")
        best_params = None
        best_clf = None

        for clf_name, clf_and_params in clf_param_dict.items():
            clf = clf_and_params['classifier']
            param_grid = clf_and_params['params']
            for params in ParameterGrid(param_grid):
                clf.set_params(**params)
                pipeline = Pipeline([
                    ('classifier', MultiOutputClassifier(clf))
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)
                val_score = f1_score(y_pred, y_val, average='macro')  # Or another metric
                print(f"{clf_name}: {val_score}")

                if val_score > best_score:
                    best_score = val_score
                    best_params = params
                    best_clf = clf_name

        # Train final model on combined train + val with the best classifier and hyperparameters
        X_train_val = np.vstack((X_train, X_val))
        y_train_val = np.vstack((y_train, y_val))
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val)
        X_test = scaler.transform(X_test)

        final_clf = classifiers_with_params[best_clf]['classifier']
      
        final_clf.set_params(**best_params)
        pipeline = Pipeline([
            ('classifier', MultiOutputClassifier(final_clf))
        ])
        pipeline.fit(X_train_val, y_train_val)

        # Test set evaluation
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        # Iterate over each symptom (output)
        for i, symptom in enumerate(y_test.columns):
            print(f"Computing SHAP values for symptom {symptom}...")

            # Get the individual classifier for the current output
            individual_clf = pipeline.named_steps['classifier'].estimators_[i]

            explainer = get_shap_explainer(individual_clf, X_test, background_data=None)

            # Compute SHAP values
            shap_values = explainer.shap_values(X_test)

            # Append SHAP values from this fold to the list for this symptom
            shap_values_per_output_all_folds[symptom].append(shap_values)
 

        # Store results for this fold
        print(best_clf)
        results_fold = {
            "fold": n_fold,
            "best_clf": best_clf,
            "best_params": best_params,
            "y_test": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        # Increment fold counter
        n_fold += 1
        results.append(results_fold)
     

    # Combine SHAP values and \( X_{\text{test}} \) across all folds
    for symptom, shap_values_list in shap_values_per_output_all_folds.items():
    #    # Flatten the list of SHAP values across all folds for this symptom
        all_shap_values = np.concatenate(shap_values_list, axis=0)
        
        # Generate SHAP summary plot for the aggregated SHAP values
        shap.summary_plot(all_shap_values, all_X_test.values, feature_names=feature_names,show=False)
        plt.title(f'Aggregated SHAP Summary for {symptom}')
        plt.savefig(f'shap_summary_groupKFold_aggregated_symptom_{symptom}.png')
        plt.close()
       # '''

    return results

def save_results_emb(fold, model_name, fusion, train_metadata_loader, val_metadata_loader, test_metadata_loader, train_embeddings, val_embeddings, test_embeddings):

    embedding_data = {
        "fold": fold,
        "model": model_name,
        "fusion": fusion,
        "train_embeddings": train_embeddings,
        "val_embeddings": val_embeddings,
        "test_embeddings": test_embeddings,
        "train_metadata": train_metadata_loader,
        "val_metadata": val_metadata_loader,
        "test_metadata": test_metadata_loader
    }
    with open(f"{BASE_DIR}/results_autoencoder_fusion/pkl/embeddings_{model_name}_{fusion}_{fold}.pkl", "wb") as f:
        pickle.dump(embedding_data, f)

def save_results_clf(fold, model_name, fusion,results):

    clf_data ={
        "fold": fold,
        "model": model_name,
        "fusion": fusion,
        "results": results
    }
    with open(f"{BASE_DIR}/results_autoencoder_fusion_clf/pkl/clf_{model_name}_{fusion}_{fold}.pkl", "wb") as f:
        pickle.dump(clf_data, f)

def save_results_binary_clf(fold, model_name, fusion, clf_name, params,results):

    clf_data ={
        "fold": fold,
        "model": model_name,
        "fusion": fusion,
        "clf": clf_name,
        "params": params,
        "results": results
    }
    with open(f"results_autoencoder_fusion_binaryclf/pkl/clf_{model_name}_{fusion}_{clf_name}_{params}_{fold}.pkl", "wb") as f:
        pickle.dump(clf_data, f)

