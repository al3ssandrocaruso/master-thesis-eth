# Set the cluster flag: 0 for local, 1 for cluster
cluster = 1  # Change this to 1 for cluster paths

if cluster == 0:
    # Local paths
    ROOT_PATH = "/Volumes/green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features_aligned"
    BASE_DIR = "/Users/crisgallego/Desktop/results_output"
    #ROOT_PATH = "/Users/crisgallego/Desktop/SMART_derived_features_aligned"
    ROOT_FOLDER = "/Users/crisgallego/Desktop/SMART_deepRLearning/data_files"
    PATH_bdi = "/Users/crisgallego/Desktop/SMART_deepRLearning/data_files/SMART_bdi_allparticipants.csv"
    PATH_DEMOGRAPHIC = "/Users/crisgallego/Desktop/SMART_deepRLearning/data_files/SMART_all_demographics.csv"
    PATH_EMA = '/Volumes/green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features/questionnaires/SMART_momentary_questions_allparticipants.csv'
    PATH_part = "/Users/alessandrocaruso/Desktop/Master Thesis Files/participants.csv"
else:
    # Cluster paths
    BASE_DIR = "/cluster/home/cgallego/results_output"
    ROOT_PATH = "/cluster/dataset/smslab/ambizione/SMART_derived_features_aligned"
    ROOT_FOLDER = "/cluster/dataset/smslab/ambizione/SMART_derived_features_aligned/data_files"
    PATH_bdi = '/cluster/dataset/smslab/ambizione/SMART_derived_features_aligned/data_files/SMART_bdi_allparticipants.csv'
    PATH_DEMOGRAPHIC = "/cluster/dataset/smslab/ambizione/SMART_derived_features_aligned/data_files/SMART_all_demographics.csv"
    PATH_EMA = '/cluster/dataset/smslab/ambizione/SMART_derived_features/questionnaires/SMART_momentary_questions_allparticipants.csv'
    PATH_part = "/Users/alessandrocaruso/Desktop/Master Thesis Files/participants.csv"

hrv_feat_dict = {'hrv_1': 'HRV_MeanNN', 'hrv_2': 'HRV_SDNN', 'hrv_3': 'HRV_SDANN1', 'hrv_4': 'HRV_SDNNI1', 'hrv_5': 'HRV_RMSSD', 'hrv_6': 'HRV_SDSD', 'hrv_7': 'HRV_CVNN', 'hrv_8': 'HRV_CVSD', 'hrv_9': 'HRV_MedianNN', 'hrv_10': 'HRV_MadNN', 'hrv_11': 'HRV_MCVNN', 'hrv_12': 'HRV_IQRNN', 'hrv_13': 'HRV_SDRMSSD', 'hrv_14': 'HRV_Prc20NN', 'hrv_15': 'HRV_Prc80NN', 'hrv_16': 'HRV_pNN50', 'hrv_17': 'HRV_pNN20', 'hrv_18': 'HRV_MinNN', 'hrv_19': 'HRV_MaxNN', 'hrv_20': 'HRV_HTI', 'hrv_21': 'HRV_TINN', 'hrv_22': 'HRV_VLF', 'hrv_23': 'HRV_LF', 'hrv_24': 'HRV_HF', 'hrv_25': 'HRV_VHF', 'hrv_26': 'HRV_TP', 'hrv_27': 'HRV_LFHF', 'hrv_28': 'HRV_LFn', 'hrv_29': 'HRV_HFn', 'hrv_30': 'HRV_LnHF', 'hrv_31': 'HRV_SD1', 'hrv_32': 'HRV_SD2', 'hrv_33': 'HRV_SD1SD2', 'hrv_34': 'HRV_S', 'hrv_35': 'HRV_CSI', 'hrv_36': 'HRV_CVI', 'hrv_37': 'HRV_CSI_Modified', 'hrv_38': 'HRV_PIP', 'hrv_39': 'HRV_IALS', 'hrv_40': 'HRV_PSS', 'hrv_41': 'HRV_PAS', 'hrv_42': 'HRV_GI', 'hrv_43': 'HRV_SI', 'hrv_44': 'HRV_AI', 'hrv_45': 'HRV_PI', 'hrv_46': 'HRV_C1d', 'hrv_47': 'HRV_C1a', 'hrv_48': 'HRV_SD1d', 'hrv_49': 'HRV_SD1a', 'hrv_50': 'HRV_C2d', 'hrv_51': 'HRV_C2a', 'hrv_52': 'HRV_SD2d', 'hrv_53': 'HRV_SD2a', 'hrv_54': 'HRV_Cd', 'hrv_55': 'HRV_Ca', 'hrv_56': 'HRV_SDNNd', 'hrv_57': 'HRV_SDNNa', 'hrv_58': 'HRV_DFA_alpha1', 'hrv_59': 'HRV_MFDFA_alpha1_Width', 'hrv_60': 'HRV_MFDFA_alpha1_Peak', 'hrv_61': 'HRV_MFDFA_alpha1_Mean', 'hrv_62': 'HRV_MFDFA_alpha1_Max', 'hrv_63': 'HRV_MFDFA_alpha1_Delta', 'hrv_64': 'HRV_MFDFA_alpha1_Asymmetry', 'hrv_65': 'HRV_MFDFA_alpha1_Fluctuation', 'hrv_66': 'HRV_MFDFA_alpha1_Increment', 'hrv_67': 'HRV_DFA_alpha2', 'hrv_68': 'HRV_MFDFA_alpha2_Width', 'hrv_69': 'HRV_MFDFA_alpha2_Peak', 'hrv_70': 'HRV_MFDFA_alpha2_Mean', 'hrv_71': 'HRV_MFDFA_alpha2_Max', 'hrv_72': 'HRV_MFDFA_alpha2_Delta', 'hrv_73': 'HRV_MFDFA_alpha2_Asymmetry', 'hrv_74': 'HRV_MFDFA_alpha2_Fluctuation', 'hrv_75': 'HRV_MFDFA_alpha2_Increment', 'hrv_76': 'HRV_ApEn', 'hrv_77': 'HRV_SampEn', 'hrv_78': 'HRV_ShanEn', 'hrv_79': 'HRV_FuzzyEn', 'hrv_80': 'HRV_MSEn', 'hrv_81': 'HRV_CMSEn', 'hrv_82': 'HRV_RCMSEn', 'hrv_83': 'HRV_CD', 'hrv_84': 'HRV_HFD', 'hrv_85': 'HRV_KFD', 'hrv_86': 'HRV_LZC', 'hrv_87': 'sqi_avg'}

hrv_columns = [
    'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_RMSSD', 'HRV_SDSD',
    'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
    'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50',
    'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN',
    'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_TP', 'HRV_LFHF',
    'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2',
    'HRV_S', 'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP',
    'HRV_IALS', 'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI',
    'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d',
    'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd',
    'HRV_SDNNa', 'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width',
    'HRV_MFDFA_alpha1_Peak', 'HRV_MFDFA_alpha1_Mean',
    'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Delta',
    'HRV_MFDFA_alpha1_Asymmetry', 'HRV_MFDFA_alpha1_Fluctuation',
    'HRV_MFDFA_alpha1_Increment', 'HRV_DFA_alpha2',
    'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak',
    'HRV_MFDFA_alpha2_Mean', 'HRV_MFDFA_alpha2_Max',
    'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry',
    'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment',
    'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn',
    'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC',
    'sqi_avg'
]

columns_to_drop = ['HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_ULF']

# Selected features based on Pearson correlation
selected_features_depression = [
    "HRV_ShanEn",
    "HRV_pNN50",
    "HRV_HTI",
    "HRV_pNN20",
    "HRV_MadNN",
    "HRV_CVI",
    "HRV_Prc80NN",
    "HRV_IQRNN",
    "HRV_MeanNN",
    "HRV_MCVNN",
    "HRV_MedianNN",
    "HRV_HFD",
    "HRV_MFDFA_alpha1_Width",
    "HRV_Prc20NN",
    "HRV_CSI",
    "HRV_SDRMSSD",
    "HRV_MFDFA_alpha1_Max",
    "HRV_SD1SD2",
    "HRV_DFA_alpha1",
    "HRV_PI",
    "HRV_TINN",
    "HRV_C2a",
    "HRV_C2d",
    "HRV_LZC",
    "HRV_HFn"
]
'''
selected_features_pearson = [
    'HRV_ShanEn',
 'HRV_pNN50',
 'HRV_MFDFA_alpha1_Width',
 'HRV_MFDFA_alpha1_Max',
 'HRV_PI',
 'HRV_TINN',
 'HRV_C2a',
 'HRV_LZC',
 'HRV_SDANN1',
 'HRV_VLF',
 'HRV_LF',
 'HRV_VHF',
 'HRV_LFn',
 'HRV_PIP',
 'HRV_PAS',
 'HRV_GI',
 'HRV_MFDFA_alpha1_Peak',
 'HRV_MFDFA_alpha1_Mean',
 'HRV_MFDFA_alpha1_Delta',
 'HRV_MFDFA_alpha1_Fluctuation',
 'HRV_DFA_alpha2',
 'HRV_MFDFA_alpha2_Width',
 'HRV_MFDFA_alpha2_Peak',
 'HRV_MFDFA_alpha2_Max',
 'HRV_MFDFA_alpha2_Delta',
 'HRV_MFDFA_alpha2_Asymmetry',
 'HRV_MFDFA_alpha2_Fluctuation',
 'HRV_ApEn',
 'HRV_SampEn',
 'HRV_MSEn',
 'HRV_CMSEn',
 'HRV_RCMSEn',
 'HRV_RMSSD',
 'HRV_MeanNN'
]
'''

selected_features_pearson = ['HRV_MeanNN','HRV_RMSSD','HRV_ShanEn',"HRV_pNN50","HRV_SDNN","HRV_LFHF","activity_count", "step_count", "run_walk_time", "resp_rate"]

#excluded_user_ids = {'SMART_201', 'SMART_001', 'SMART_002', 'SMART_005', 'SMART_037', 'SMART_052', 'SMART_055', 'SMART_145', 'SMART_202'}
excluded_user_ids = {'SMART_001', 'SMART_002', 'SMART_005', 'SMART_037','SMART_055'}

social_columns = [f'bt_{i+1}' for i in range(1,2)] + [f'gps_{i+1}' for i in range(8)]

keys = {"physiological": ["user", "day", "hour", "activity_count", "step_count", "run_walk_time", "resp_rate", "hrv_data"], "social":["user", "day", "hour", "bt_data", "gps_data"]}

models = ['lstm', 'cnn', 'rnn', 'transformer','lstm_vae', 'cnn_vae','rnn_vae','transformer_vae']

#models = ['cnn_vae']
embedding_size = [32,64,128,256,512]

num_hidden_layers = [1,2]

fusion_types = ['early']#, 'mid', 'late']
