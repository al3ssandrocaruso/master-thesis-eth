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

def extract_embeddings(model, dataloader_emb, device, input_dims):
    physiological_input_dim, _ = input_dims
    model.eval()
    embeddings, metadata_loader = [], []

    with torch.no_grad():
        for sequences, meta in dataloader_emb:
            sequences = sequences.to(device)
            sequences_physiological = sequences[:, :, :physiological_input_dim]
            sequences_social = sequences[:, :, physiological_input_dim:]
            try:
                _, embedding = model(sequences_social, sequences_physiological)
            except:
                _,_,_,_,embedding = model(sequences_social, sequences_physiological)
            metadata_loader.append(meta)
            embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings), np.vstack(metadata_loader)
           
def main():

    ## NO 10-fold CV

    args = get_args()
    print_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_directories()
    
    loso = 'noCV'
    folds = load_folds(args,loso)

    physiological_input_dim = len(selected_features_pearson)
    social_input_dim = len(social_columns)

    expanded_bdi = load_prepare_bdi()
    df_participants = pd.read_csv(PATH_DEMOGRAPHIC)

    model = args.model
    for fusion in fusion_types:
        n_fold = 0
        for fold in folds:
            print(len(folds))
            dataloader_train, dataloader_val, dataloader_test, _, _, max_seq_length = fold
            print(f"Training {model} fusion type {fusion}")
            autoencoder = initialize_model(
                model, fusion, (social_input_dim, physiological_input_dim), args.hidden_dim, args.embedding_dim,
                max_seq_length, device, args.intermediate_size, args.num_hidden_layers, args.num_attention_heads
            )

            if 'vae' in model: 
                vae = True
            else:
                vae = False

            criterion = nn.MSELoss()
            optimizer = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
            trained_model = train_model(autoencoder, dataloader_train, dataloader_val, criterion, optimizer, args.num_epochs, device, (physiological_input_dim, social_input_dim),vae)
            embeddings_train, metadata_train = extract_embeddings(trained_model, dataloader_train, device, (physiological_input_dim, social_input_dim))
            embeddings_val, metadata_val = extract_embeddings(trained_model, dataloader_val, device, (physiological_input_dim, social_input_dim))
            embeddings_test, metadata_test = extract_embeddings(trained_model, dataloader_test, device, (physiological_input_dim, social_input_dim))
            
            save_results_emb(n_fold, model, fusion, metadata_train, metadata_val, metadata_test, embeddings_train, embeddings_val, embeddings_test)

            results_multi = perform_classification(classifiers_with_params, embeddings_train, metadata_train, embeddings_val, metadata_val, embeddings_test, metadata_test, expanded_bdi, df_participants, demo=False, mood=False)
            
            save_results_clf(n_fold, model,fusion,results_multi)

            n_fold += 1

if __name__ == "__main__":
    main()
