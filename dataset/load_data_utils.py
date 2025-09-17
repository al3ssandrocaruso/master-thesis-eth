from config.config import keys, social_columns
import warnings
warnings.filterwarnings('ignore')
import warnings
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from config.config import selected_features_pearson, hrv_feat_dict, PATH_bdi, BASE_DIR

from dataset.dataset import UserData, TimeSeriesDataset
import pickle
from utils.datetime.datetime_utils import map_hour_to_windows
from utils.questionnaires.bdi import expand_bdi_data
warnings.filterwarnings("ignore")

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
        kf = GroupKFold(n_splits=10,random_state=42)
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
    print(len(folds))
    return folds

def extract_features(data, feature_name, feature_prefix):
    """
    Extract features from a nested list in the dataframe and expand them into separate columns.

    Args:
    - data (pd.DataFrame): The dataframe containing the nested list data.
    - feature_name (str): The column name of the nested list data.
    - feature_prefix (str): The prefix for the new column names.

    Returns:
    - pd.DataFrame: The dataframe with the expanded features.
    """
    columns = [f'{feature_prefix}_{i + 1}' for i in range(len(data[feature_name][0]))]
    expanded_features = pd.DataFrame(data[feature_name].tolist(), index=data.index, columns=columns)
    return expanded_features


def load_data(modality):
    """
    Load user data based on the specified modality and process it to extract relevant features.

    Args:
    - modality (str): The modality of the data to load ('physiological', 'social', or 'multi-view').

    Returns:
    - X (pd.DataFrame): The feature dataframe after selection and processing.
    - data (pd.DataFrame): The original dataframe with additional processed features.
    """
    # Determine the flag for social data usage
    flag = (modality == 'social') or (modality == 'multi-view')
    samples = UserData(cluster=False, use_social_data=flag)

    if modality == 'physiological':
        selected_keys = keys['physiological']
    elif modality == 'social':
        selected_keys = keys['social']
    elif modality == 'multi-view':
        selected_keys = keys['physiological'] + keys['social']

    feature_name, feature_prefix = ('hrv_data', 'hrv') if modality in ['physiological', 'multi-view'] else (None, None)

    # Filter the samples based on selected keys
    data = [{k: v for k, v in sample.items() if k in selected_keys} for sample in samples]
    df = pd.DataFrame(data)

    # Process physiological data if present
    if modality in ['physiological', 'multi-view']:
        expanded_features = extract_features(df, feature_name, feature_prefix)
        expanded_features.columns = [hrv_feat_dict[col] for col in expanded_features.columns]
        df = df.drop(columns=feature_name).join(expanded_features)

    # Process social data if present
    if modality in ['social', 'multi-view']:
        bt_features = extract_features(df, 'bt_data', 'bt')
        gps_features = extract_features(df, 'gps_data', 'gps')
        df = df.drop(columns=['bt_data', 'gps_data']).join(bt_features).join(gps_features)
        df.drop(columns=['bt_1'], inplace=True)

    # Ensure 'day' is in datetime format
    df['day'] = pd.to_datetime(df['day'])
    data = df.copy()

    # Select features based on Pearson correlation for physiological data
    if modality == 'physiological':
        X = data[selected_features_pearson]
    else:
        X = data.iloc[:, 3:]

    X.fillna(0, inplace=True)

    return X, data


def load_data_embeddings(use_social_data=False):
    samples = UserData(use_social_data=use_social_data)
    depressed_users, healthy_users, _, _ = samples.get_partecipants_info()

    if  use_social_data:
        data = [
            {k: v for k, v in i.items() if
             k in ["user", "day", "hour", "activity_count", "step_count", "run_walk_time", "resp_rate", "hrv_data","bt_data", "gps_data"]}
            for i in samples
        ]
    else:
        data = [
            {k: v for k, v in i.items() if k in ["user", "day", "hour", "activity_count", "step_count", "run_walk_time", "resp_rate", "hrv_data"]}
            for i in samples
        ]

    df = pd.DataFrame(data)
    hrv_columns = [hrv_feat_dict[f'hrv_{i + 1}'] for i in range(len(df['hrv_data'][0]))]
    df[hrv_columns] = pd.DataFrame(df['hrv_data'].tolist(), index=df.index)
    df.drop('hrv_data', axis=1, inplace=True)

    if use_social_data:
        bt_columns = [f'bt_{i + 1}' for i in range(len(df['bt_data'][0]))]
        df[bt_columns] = pd.DataFrame(df['bt_data'].tolist(), index=df.index)
        df.drop('bt_data', axis=1, inplace=True)
        # df.drop('bt_1', axis=1, inplace=True)

        gps_columns = [f'gps_{i + 1}' for i in range(len(df['gps_data'][0]))]
        df[gps_columns] = pd.DataFrame(df['gps_data'].tolist(), index=df.index)
        df.drop('gps_data', axis=1, inplace=True)

    df.dropna(inplace=True)
    df['hh'] = pd.to_datetime(df['hour']).dt.hour
    return df, depressed_users, healthy_users


def expand_rows(df, interval_length, overlap):
    expanded_rows = []
    num_windows = 24 // (interval_length - overlap)
    for index, row in df.iterrows():
        hour = row['hh']
        windows = map_hour_to_windows(hour, interval_length, overlap)
        for window in windows:
            new_row = row.copy()
            new_row['window'] = window
            expanded_rows.append(new_row)
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df['day'] = pd.to_datetime(expanded_df['day'])
    expanded_df.loc[(expanded_df['window'] == num_windows) & (expanded_df['hh'] >= 0) & (expanded_df['hh'] <= overlap), 'day'] -= pd.Timedelta(days=1)
    expanded_df['day'] = pd.to_datetime(expanded_df['day']).dt.strftime('%Y-%m-%d')
    expanded_df.drop('hh', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    return expanded_df


def group_and_filter_data(df, threshold=0.7):
    grouped_df = df.groupby(['user', 'day', 'window']).apply(lambda x: x.sort_values('hour')).reset_index(drop=True)
    grouped_df.drop('hour', axis=1, inplace=True)
    group_counts = grouped_df.groupby(['user', 'day', 'window']).size()
    max_length = group_counts.max()
    grouped_dfs = {name: group for name, group in grouped_df.groupby(['user', 'day', 'window'])}
    grouped_dfs = {name: group for name, group in grouped_dfs.items() if group_counts[name] >= max_length * threshold}
    return grouped_dfs

def normalize_features(grouped_dfs, scaler_type,  multimodal_autoencoder=False):
    if multimodal_autoencoder:
        all_features = pd.concat([group[selected_features_pearson + social_columns] for group in grouped_dfs.values()])
    else:
        all_features = pd.concat([group[selected_features_pearson] for group in grouped_dfs.values()])
    scaler = MinMaxScaler() if scaler_type == 'MinMax' else StandardScaler()
    scaler.fit(all_features)
    for name, group in grouped_dfs.items():
        index_df = group[['user', 'day', 'window']]
        features = group[selected_features_pearson + social_columns] if multimodal_autoencoder else group[selected_features_pearson]
        normalized_features_array = scaler.transform(features)
        # normalized_features_array = features
        normalized_features_df = pd.DataFrame(normalized_features_array, index=features.index, columns=features.columns)
        grouped_dfs[name] = pd.concat([index_df, normalized_features_df], axis=1)
    return grouped_dfs

def prepare_sequences(grouped_dfs):
    sequences = []
    metadata = []
    for name, group in grouped_dfs.items():
        sequences.append(group.drop(columns=['user', 'day', 'window']).values)
        metadata.append(group[['user', 'day', 'window']].iloc[0].values)
    return sequences, metadata

def pad_sequences(sequences):
    max_seq_length = max([len(seq) for seq in sequences])
    padded_sequences = []
    for seq in sequences:
        mean_values = np.mean(seq, axis=0)
        padding_length = max_seq_length - len(seq)
        if padding_length > 0:
            padding = np.tile(mean_values, (padding_length, 1))
            padded_seq = np.vstack((seq, padding))
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

def custom_collate(batch):
    sequences, metadata = zip(*batch)
    sequences = torch.stack(sequences)
    return sequences, metadata

def create_dataloader(data_tensor, metadata, batch_size = 32):
    dataset = TimeSeriesDataset(data_tensor, metadata)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

def get_data_loader_noCV(interval_length, overlap, scaler, multimodal_autoencoder=False, return_user_lists=False, batch_size = 16):
    # Load and preprocess the data
    folds = []

    df, depressed_users, healthy_users = load_data_embeddings(multimodal_autoencoder)
    df = expand_rows(df, interval_length, overlap)
    grouped_dfs = group_and_filter_data(df)
    grouped_dfs = normalize_features(grouped_dfs, scaler, multimodal_autoencoder)

    # Prepare sequences and pad them
    sequences, metadata = prepare_sequences(grouped_dfs)
    data = pad_sequences(sequences)

    # Convert data to tensor format
    data_tensor = torch.tensor(data, dtype=torch.float32)
    metadata = [list(item) for item in metadata]

    # Create dataloader
    dataloader = create_dataloader(data_tensor, metadata, batch_size)
    input_dim = data_tensor.shape[2]

    if multimodal_autoencoder:
        # Additional dataloader for multimodal autoencoder
        dataloader_emb = create_dataloader(data_tensor, metadata, batch_size=1)
        max_seq_length = max(len(seq) for seq in sequences)
        folds.append((dataloader, dataloader, dataloader, input_dim, dataloader_emb, max_seq_length))
    else:
        folds.append((dataloader, dataloader, dataloader, input_dim))

    if return_user_lists:
        return folds, depressed_users, healthy_users
    else:
        return folds

def get_data_loader_loso(interval_length, overlap, scaler, multimodal_autoencoder=False, return_user_lists=False, batch_size = 16):
    
    # Load and preprocess the data
    df, depressed_users, healthy_users = load_data_embeddings(multimodal_autoencoder)
    df = expand_rows(df, interval_length, overlap)
    grouped_dfs = group_and_filter_data(df)
    grouped_dfs = normalize_features(grouped_dfs, scaler, multimodal_autoencoder)

    # Prepare sequences and pad them
    sequences, metadata = prepare_sequences(grouped_dfs)
    data = pad_sequences(sequences)
    groups = [item[0] for item in metadata]

    # Convert data to tensor format
    data_tensor = torch.tensor(data, dtype=torch.float32)
    metadata = [list(item) for item in metadata]
    
    total_size = len(data)
    indices = list(range(total_size))
    
    kf = GroupKFold(n_splits=10)
    indices = np.arange(len(data_tensor))  # Assuming `indices` are just the range of dataset size
    folds = []

    # Create the KFold splits once
    all_folds = list(kf.split(indices,groups=groups))

    for test_fold_idx in range(len(all_folds)):
        # Test fold
        _, test_indices = all_folds[test_fold_idx]
        test_set = Subset(data_tensor, test_indices)
        metadata_test = Subset(metadata, test_indices)
        dataloader_test = create_dataloader(test_set, metadata_test, batch_size)
        
        # Validation fold (cyclically pick the next fold)
        val_fold_idx = (test_fold_idx + 1) % len(all_folds)
        _, val_indices = all_folds[val_fold_idx]
        val_set = Subset(data_tensor, val_indices)
        metadata_val = Subset(metadata, val_indices)
        dataloader_val = create_dataloader(val_set, metadata_val, batch_size)

        # Training folds (all other folds except test and validation)
        train_indices = []
        for fold_idx, (_, train_fold) in enumerate(all_folds):
            if fold_idx != test_fold_idx and fold_idx != val_fold_idx:
                train_indices.extend(train_fold)
        
        train_set = Subset(data_tensor, train_indices)
        metadata_train = Subset(metadata, train_indices)
        dataloader_train = create_dataloader(train_set, metadata_train, batch_size)
        
        input_dim = data_tensor.shape[2]

        if multimodal_autoencoder:
            # Additional dataloader for multimodal autoencoder
            dataloader_emb = create_dataloader(data_tensor, metadata, batch_size=1)
            max_seq_length = max(len(seq) for seq in sequences)
            folds.append((dataloader_train, dataloader_val, dataloader_test, input_dim, dataloader_emb, max_seq_length))
        else:
            folds.append((dataloader_train, dataloader_val, dataloader_test, input_dim))

    if return_user_lists:
        return folds, depressed_users, healthy_users
    return folds



def get_data_loader(interval_length, overlap, scaler, multimodal_autoencoder=False, return_user_lists=False, batch_size = 16):
    
    # Load and preprocess the data
    df, depressed_users, healthy_users = load_data_embeddings(multimodal_autoencoder)
    df = expand_rows(df, interval_length, overlap)
    grouped_dfs = group_and_filter_data(df)
    grouped_dfs = normalize_features(grouped_dfs, scaler, multimodal_autoencoder)

    # Prepare sequences and pad them
    sequences, metadata = prepare_sequences(grouped_dfs)
    data = pad_sequences(sequences)

    # Convert data to tensor format
    data_tensor = torch.tensor(data, dtype=torch.float32)
    metadata = [list(item) for item in metadata]
    
    total_size = len(data)
    indices = list(range(total_size))
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    indices = np.arange(len(data_tensor))  # Assuming `indices` are just the range of dataset size
    folds = []

    # Create the KFold splits once
    all_folds = list(kf.split(indices))

    for test_fold_idx in range(len(all_folds)):
        # Test fold
        _, test_indices = all_folds[test_fold_idx]
        test_set = Subset(data_tensor, test_indices)
        metadata_test = Subset(metadata, test_indices)
        dataloader_test = create_dataloader(test_set, metadata_test, batch_size)
        
        # Validation fold (cyclically pick the next fold)
        val_fold_idx = (test_fold_idx + 1) % len(all_folds)
        _, val_indices = all_folds[val_fold_idx]
        val_set = Subset(data_tensor, val_indices)
        metadata_val = Subset(metadata, val_indices)
        dataloader_val = create_dataloader(val_set, metadata_val, batch_size)

        # Training folds (all other folds except test and validation)
        train_indices = []
        for fold_idx, (_, train_fold) in enumerate(all_folds):
            if fold_idx != test_fold_idx and fold_idx != val_fold_idx:
                train_indices.extend(train_fold)
        
        train_set = Subset(data_tensor, train_indices)
        metadata_train = Subset(metadata, train_indices)
        dataloader_train = create_dataloader(train_set, metadata_train, batch_size)
        
        input_dim = data_tensor.shape[2]

        if multimodal_autoencoder:
            # Additional dataloader for multimodal autoencoder
            dataloader_emb = create_dataloader(data_tensor, metadata, batch_size=1)
            max_seq_length = max(len(seq) for seq in sequences)
            folds.append((dataloader_train, dataloader_val, dataloader_test, input_dim, dataloader_emb, max_seq_length))
        else:
            folds.append((dataloader_train, dataloader_val, dataloader_test, input_dim))

    if return_user_lists:
        return folds, depressed_users, healthy_users
    return folds


def save_folds(folds):

    with open(f"{BASE_DIR}/intermediate_data/pkl/folds_10cv.pkl", "wb") as f:
        pickle.dump(folds, f)

def load_folds(args, loso, filepath=f"{BASE_DIR}/intermediate_data/pkl/folds_10cv.pkl"):
    #try:
    #    with open(filepath, "rb") as f:
    #        folds = pickle.load(f)
        
    #except:
    if loso==True:
        folds = get_data_loader_loso(
            args.interval_length, args.overlap, args.scaler, multimodal_autoencoder=True
    )
    elif loso== 'noCV':
        folds = get_data_loader_noCV(
            args.interval_length, args.overlap, args.scaler, multimodal_autoencoder=True
    )
    else:
        folds = get_data_loader(
            args.interval_length, args.overlap, args.scaler, multimodal_autoencoder=True
        )
    save_folds(folds)

    return folds

def load_prepare_bdi():
    # Load and prepare BDI
    df_bdi = pd.read_csv(PATH_bdi)
    df_bdi['bdi_time'] = pd.to_datetime(df_bdi['bdi_time'])
    df_bdi['day'] = df_bdi['bdi_time'].dt.date
    df_bdi = df_bdi.drop(columns=['bdi_time'])
    df_bdi.rename(columns={'study_id': 'user'}, inplace=True)
    expanded_bdi = expand_bdi_data(df_bdi)
    
    return expanded_bdi
    



