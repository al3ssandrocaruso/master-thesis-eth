import warnings
import numpy as np
import pandas as pd
from config.config import *
from dataset.load_data_utils import load_prepare_bdi
from utils.model_training.training_utils import *
from itertools import product
from utils.classification.classification_utils import *
warnings.filterwarnings("ignore")

def main():

    expanded_bdi = load_prepare_bdi()
    df_participants = pd.read_csv(PATH_DEMOGRAPHIC)

    results = perform_classification_ema(classifiers_with_params, expanded_bdi, df_participants=None)
    save_results_clf('loso', 'ema', 'noDemo',results)

if __name__ == "__main__":
    main()