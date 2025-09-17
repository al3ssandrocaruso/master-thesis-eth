import pandas as pd
from config.config import hrv_feat_dict
from dataset.dataset import UserData
from scipy.stats import pearsonr, spearmanr
import warnings

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Load user data
    dataset = UserData()

    # Create DataFrame from dataset
    df = pd.DataFrame([data for data in dataset])

    # Extract HRV features
    hrv_columns = [f'hrv_{i + 1}' for i in range(len(df['hrv_data'][0]))]
    df[hrv_columns] = pd.DataFrame(df['hrv_data'].tolist(), index=df.index)

    # Drop unnecessary columns
    df.drop(['hrv_data', 'day', 'hour'], axis=1, inplace=True)

    # Load participant data and merge with HRV data
    df_part = pd.read_csv("/Users/alessandrocaruso/Desktop/Master Thesis Files/participants.csv")
    df_part.rename(columns={'key_smart_id': 'user'}, inplace=True)
    df_part = df_part[['user', 'type']]
    df = pd.merge(df_part, df, on='user')
    df.drop('user', axis=1, inplace=True)

    # Rename HRV feature columns
    df.rename(columns=hrv_feat_dict, inplace=True)

    # Create depression label
    df['depression_label'] = df['type'].apply(lambda x: 1 if x == 'p' else 0)
    df.drop(columns=['type'], inplace=True)
    df.dropna(inplace=True)

    # Calculate Pearson and Spearman correlations
    correlation_scores_pearson = {}
    correlation_scores_spearman = {}

    feature_names = df.columns[:-1]

    for feature in feature_names:
        pearson_corr, _ = pearsonr(df[feature], df['depression_label'])
        spearman_corr, _ = spearmanr(df[feature], df['depression_label'])
        correlation_scores_pearson[feature] = abs(pearson_corr)
        correlation_scores_spearman[feature] = abs(spearman_corr)

    # Sort features based on their correlation scores
    sorted_features_pearson = sorted(correlation_scores_pearson.items(), key=lambda x: x[1], reverse=True)
    sorted_features_spearman = sorted(correlation_scores_spearman.items(), key=lambda x: x[1], reverse=True)

    # Print sorted features by importance
    print("Selected Features (Pearson):")
    for feature, score in sorted_features_pearson:
        print(f"{feature}: {score:.4f}")

    print("\nSelected Features (Spearman):")
    for feature, score in sorted_features_spearman:
        print(f"{feature}: {score:.4f}")

if __name__ == "__main__":
    main()