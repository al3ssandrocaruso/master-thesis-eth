from datetime import timedelta
from config.config import PATH_bdi, PATH_part
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def map_hamd_score(score):
    if 0 <= score <= 3:
        return 0
    elif 4 <= score <= 12:
        return 1
    elif 13 <= score <= 18:
        return 2
    else:
        return 4

def read_file(pickle_path):
    pickle = pd.read_pickle(pickle_path)
    metadata = pickle["metadata"]
    embeddings = pickle["embeddings"]
    return embeddings, metadata

def get_hdrs_labels():
    # Read the HDRS CSV file
    hdrs = pd.read_csv("/Users/alessandrocaruso/Desktop/HDRS_allusers.csv")
    hdrs = hdrs[(hdrs['redcap_event_name'] == 'intervention_1__la_arm_1') | (hdrs['redcap_event_name'] == 'intervention_2__la_arm_1')]
    hdrs_labels = hdrs[['study_id','date','hamd_17_score']]
    hdrs_labels['date'] = pd.to_datetime(hdrs_labels['date'], format='%d.%m.%y').dt.strftime('%Y-%m-%d')
    hdrs_labels['hamd_label'] = hdrs_labels['hamd_17_score'].apply(map_hamd_score)
    hdrs_labels = hdrs_labels.drop(['hamd_17_score'], axis=1)

    # Remove duplicates if any
    hdrs_labels = hdrs_labels.drop_duplicates(subset=['study_id', 'date'])

    # Read embeddings and metadata
    file_path = "/Users/alessandrocaruso/Documents/RESULTS/results_autoencoder[8h_4o]_64/embedding_data_bert_autoencoder.pkl"
    embeddings, metadata = read_file(file_path)
    metadata = np.array(metadata)[:,:2]
    df_metadata = pd.DataFrame(metadata, columns=['study_id','date'])

    # Remove duplicates if any
    df_metadata = df_metadata.drop_duplicates(subset=['study_id', 'date'])

    # Merge the two DataFrames
    merged_df = pd.merge(df_metadata, hdrs_labels, on=['study_id','date'])

    merged_df['date'] = pd.to_datetime(merged_df['date'])

    # Rename columns
    merged_df.rename(columns={'study_id': 'user', 'date': 'day'}, inplace=True)

    # Expand data to cover the past 7 days
    expanded_df = pd.DataFrame()
    for _, row in merged_df.iterrows():
        for i in range(7):
            expanded_row = {
                'user': row['user'],
                'day': (row['day'] - timedelta(days=i)).strftime('%Y-%m-%d'),
                'hamd_label': row['hamd_label']
            }
            expanded_df = expanded_df.append(expanded_row, ignore_index=True)

    return expanded_df

