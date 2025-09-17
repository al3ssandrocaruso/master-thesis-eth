import warnings
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
from config.config import *
from dataset.load_data_utils import load_folds, load_prepare_bdi
from utils.model_training.training_utils import *
from itertools import product
from utils.classification.classification_utils import *
warnings.filterwarnings("ignore")

torch.manual_seed(0)

def save_classification(results,model_name,fusion,folder):
    path = '/Users/crisgallego/Desktop/SMART_deepRLearning/results_all_experiments/classification/multimodal_emb/' + folder
    with open(f"{path}/clusters_{model_name}_{fusion}.pkl", "wb") as f:
        pickle.dump(results, f)

def main():
    emb_folder = '/Users/crisgallego/Desktop/SMART_deepRLearning/results_all_experiments/embeddings/multimodal/embeddings_128' #TODO: add folder where fold embeddings restuls from autoencoder are stored (10 pkl files)

    expanded_bdi = load_prepare_bdi()
    df_participants = pd.read_csv(PATH_DEMOGRAPHIC)  

    for root, dirs, files in os.walk(emb_folder):
        for dir_name in dirs:
            results = []
            model_name = dir_name
            dir_path = os.path.join(root, dir_name)
            fold_idx = 0
            # List files inside this specific directory
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if file_name.endswith('pkl'):
                    try:
                        with open(file_path, "rb") as f:
                            fold_data = pickle.load(f)
                    except:
                        break
                else:
                    continue

                # Load data
                embeddings_train = fold_data["train_embeddings"]
                metadata_train = fold_data['train_metadata']
                embeddings_val =  fold_data["val_embeddings"]
                metadata_val = fold_data['val_metadata']
                embeddings_test = fold_data['test_embeddings']
                metadata_test = fold_data['test_metadata']

                results_clf = perform_classification(classifiers_with_params, embeddings_train, metadata_train, embeddings_val, metadata_val, embeddings_test, metadata_test, expanded_bdi, df_participants, demo=False, mood=False)
                results_clf['fold'] = fold_idx

                results.append(results_clf)

                print(f"Fold {fold_idx}")
                fold_idx += 1

                save_classification(results,model_name,'early','embeddings_128')

if __name__ == "__main__":
    main()
