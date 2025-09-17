import os
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from config.config import *
from utils.datetime.datetime_utils import convert_to_integer
from utils.datetime.datetime_utils import convert_to_integer


# user_subset = ['SMART_013','SMART_012']
class UserData(Dataset):
        def __init__(self, filter = 'SMART_0', use_social_data = False, cluster = True):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.cluster = cluster if device == "cuda" else False
            self.use_social_data = use_social_data
            self.root_path = ROOT_PATH
            self.user_dirs = [user for user in os.listdir(self.root_path) if user.startswith(filter)  and user not in excluded_user_ids]
            # used for debugging :)
            # self.user_dirs = [user for user in os.listdir(self.root_path) if user in user_subset]
            self.samples = self._create_dataset()
            self.depressed_users,self.healthy_users,self.anti_depr_users, self.no_anti_depr_users = self.get_partecipants_info()

        def get_depressed_users(self):
            return self.depressed_users

        def get_healthy_users(self):
            return self.healthy_users

        def get_anti_depr_users(self):
            return self.anti_depr_users

        def get_no_anti_depr_users(self):
            return self.no_anti_depr_users

        def _create_dataset(self):
            dataset = []

            # Iterate users
            for user in tqdm(self.user_dirs, desc='Processing users'):
                file_path_hrv = f'{self.root_path}/{user}/hrvmetrics/{user}_hrvmetrics_winlen5_overlap_0.csv'
                file_path_acc = f'{self.root_path}/{user}/accmetrics/{user}_accmetrics_winlen_5.csv'
                file_path_rr = f'{self.root_path}/{user}/rrmetrics/{user}_rrmetrics_winlen_5.csv'

                try:
                    # Read HRV data
                    df_hrv = pd.read_csv(file_path_hrv, index_col=0)

                    # Read acceleration data
                    df_acc = pd.read_csv(file_path_acc, index_col=0)

                    # Read and align respiration rate data
                    df_rr = pd.read_csv(file_path_rr)
                    if not self.cluster:
                        df_rr['t_start_utc'] = df_rr.apply(lambda row: row['t_start_utc'][:19], axis=1)
                        df_rr['value.time'] = df_rr.apply(lambda row: convert_to_integer(row['t_start_utc']), axis=1)
                    df_rr.dropna()

                    # Join tables
                    df = pd.merge(df_acc, df_hrv, on=['value.time','t_start_utc'])
                    # excellent_rows = df[(df['sqi'] == 'Excellent') & (df['activity_counts'].notna())]
                    excellent_rows = df[(df['sqi_avg'] >= 0.75) & (df['activity_counts'].notna())]

                    excellent_rows = pd.merge(excellent_rows, df_rr, on=['value.time','t_start_utc'])

                    if self.use_social_data:
                        if self.cluster:
                            # /Users/alessandrocaruso/allusers_gps_data
                            file_path_gps = f"{ROOT_FOLDER}/allusers_gps_data/{user}/{user}_gps_data.csv"
                            file_path_bt = f"{ROOT_FOLDER}/allusers_bt_data/{user}/{user}_bluetooth_data.csv"
                        else:
                            file_path_gps = f"{ROOT_FOLDER}/allusers_gps_data/{user}/{user}_gps_data.csv"
                            file_path_bt = f"{ROOT_FOLDER}/allusers_bt_data/{user}/{user}_bluetooth_data.csv"

                        # Read data
                        df_gps = pd.read_csv(file_path_gps)
                        df_gps.dropna()
                        #df_gps = df_gps.drop(columns=['t_start_utc'])

                        df_bt = pd.read_csv(file_path_bt)
                        df_bt.dropna()
                        #df_bt = df_bt.drop(columns=['t_start_utc'])

                        excellent_rows = pd.merge(excellent_rows, df_bt, on=['value.time'])
                        excellent_rows = pd.merge(excellent_rows, df_gps, on=['value.time'])

                    for _, row in excellent_rows.iterrows():
  
                        date_time_obj = datetime.strptime(row['t_start_utc_x'], '%Y-%m-%d %H:%M:%S')
                        day = str(date_time_obj.date())
                        hour = date_time_obj.strftime('%H:%M')
                        activity_count, step_count, run_walk_time, resp_rate = row[['activity_counts', 'step_count', 'run_walk_time','resp_rate']]

                        hrv_data = row.loc[hrv_columns].values


                        if self.use_social_data:
                            bluetooth_data = row.loc[['unlock_duration','value.nearbyDevices']].values
                            gps_data = row.loc[['time_home','gyration','max_loc_home','rand_entropy','real_entropy','max_dist','nr_loc','nr_visits']].values
                            sample = (user, day, hour, activity_count, step_count, run_walk_time, resp_rate, hrv_data, bluetooth_data, gps_data)
                        else:
                            sample = (user, day, hour, activity_count, step_count, run_walk_time,resp_rate, hrv_data)

                        dataset.append(sample)

                except PermissionError as pe:
                    print(f"Permission denied reading file {file_path_hrv}: {pe}")
                    return None
                except Exception as e:
                    print(f"Error Occurred: {e} for user {user}")
                    return None

            return dataset

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            keys = ["user", "day", "hour", "activity_count", "step_count", "run_walk_time", "resp_rate", "hrv_data"]

            if self.use_social_data:
                keys.extend(["bt_data", "gps_data"])

            sample_tuple = self.samples[index]
            sample = {keys[i]: sample_tuple[i] for i in range(len(keys))}
            return sample

        def get_partecipants_info(self):
            if self.cluster:
                df_participants = pd.read_csv(PATH_DEMOGRAPHIC)
            else:
                df_participants = pd.read_csv(PATH_DEMOGRAPHIC)

            depressed_users = df_participants[df_participants['type'] == 'p']['user'].values
            healthy_users = df_participants[df_participants['type'] == 'h']['user'].values

            anti_depr_users = df_participants[df_participants['antidepressant_type'].notna()]['user'].values
            no_anti_depr_users = df_participants[df_participants['antidepressant_type'].isna()]['user'].values

            return depressed_users, healthy_users, anti_depr_users, no_anti_depr_users


class TimeSeriesDataset(Dataset):
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.metadata[idx]
