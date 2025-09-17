from datetime import timedelta

from config.config import PATH_bdi, PATH_part
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import timedelta

def prepare_bdi(path_to_csv, path_bdi = PATH_bdi, path_part = PATH_part):
    # Load data
    df_cluster = pd.read_csv(path_to_csv)
    df_bdi = pd.read_csv(path_bdi)
    df_part = pd.read_csv(path_part)

    # Merge participants data with physiological data
    df_part.rename(columns={'key_smart_id': 'user'}, inplace=True)
    df_part = df_part[['user', 'type']]
    df_cluster = pd.merge(df_cluster, df_part, on='user')

    # Prepare BDI data
    df_bdi['bdi_time'] = pd.to_datetime(df_bdi['bdi_time'])
    df_bdi['day'] = df_bdi['bdi_time'].dt.date
    df_bdi = df_bdi.drop(columns=['bdi_time'])
    df_bdi.rename(columns={'study_id': 'user'}, inplace=True)

    # Expand BDI data
    expanded_bdi = expand_bdi_data(df_bdi)

    return df_cluster, expanded_bdi

def expand_bdi_data(df_bdi):
    expanded_bdi = pd.DataFrame()
    for _, row in df_bdi.iterrows():
        for i in range(7):
            expanded_row = {
                'day': row['day'] - timedelta(days=i),
                'bdi_score': row['bdi_score'],
                'user': row['user']
            }
            for j in range(1, 22):
                expanded_row[f'bdi_{j}'] = row[f'bdi_{j}']
            expanded_bdi = pd.concat((expanded_bdi,pd.DataFrame(expanded_row,index=[0])), ignore_index=True)

    expanded_bdi['day'] = expanded_bdi['day'].astype(str)
    df_bdi['day'] = df_bdi['day'].astype(str)

    return expanded_bdi

