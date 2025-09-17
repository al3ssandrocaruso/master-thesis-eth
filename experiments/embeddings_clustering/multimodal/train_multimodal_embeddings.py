import os
import pickle
import warnings
import argparse
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from torch import optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from config.config import selected_features_pearson, social_columns
from dataset.load_data_utils import get_data_loader
from models.multimodal.multimodal_embeddings_models import GeneralizedLateFusionAutoencoder, GeneralizedMidFusionAutoencoder,GeneralizedEarlyFusionAutoencoder
from utils.model_training.training_utils import EarlyStopping
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Multimodal Embedding Training and Embedding Extraction')
    parser.add_argument('--interval_length', type=int, default=24, help='Interval length for data windows')
    parser.add_argument('--overlap', type=int, default=12, help='Overlap between data windows')
    parser.add_argument('--scaler', type=str, default='Standard', choices=['MinMax', 'Standard'], help='Scaler for preprocessing data')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of hidden layers (Default: 1)')
    parser.add_argument('--num_attention_heads', type=int, default=1, help='Number of attention heads (Default: 1)')
    parser.add_argument('--intermediate_size', type=int, default=512, help='Intermediate size for feed-forward layers (Default: 512)')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension size')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    return parser.parse_args()


def setup_directories():
    os.makedirs("results_autoencoder_fusion/csv", exist_ok=True)
    os.makedirs("results_autoencoder_fusion/img", exist_ok=True)
    os.makedirs("results_autoencoder_fusion/pkl", exist_ok=True)


def initialize_model(model_type, fusion_type, input_dims, hidden_dim, embedding_dim, max_seq_length, device, intermediate_size, num_hidden_layers, num_attention_heads):
    social_input_dim, physiological_input_dim = input_dims

    if fusion_type == 'early':
        return GeneralizedEarlyFusionAutoencoder(
            model_type, social_input_dim, physiological_input_dim, hidden_dim, embedding_dim,
            num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size
        ).to(device)

    if fusion_type == 'mid':
        return GeneralizedMidFusionAutoencoder(
            model_type, model_type, social_input_dim, physiological_input_dim, max_seq_length, hidden_dim,
            embedding_dim, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size
        ).to(device)

    if fusion_type == 'late':
        return GeneralizedLateFusionAutoencoder(
            model_type, model_type, social_input_dim, physiological_input_dim, hidden_dim, embedding_dim,
            num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size
        ).to(device)


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, input_dims):
    physiological_input_dim, _ = input_dims
    early_stopping = EarlyStopping()

    for epoch in tqdm(range(num_epochs), desc="Training.."):
        model.train()
        # Training phase
        train_loss = 0
        for sequences, _ in train_dataloader:
            sequences = sequences.to(device)
            sequences_physiological = sequences[:, :, :physiological_input_dim]
            sequences_social = sequences[:, :, physiological_input_dim:]

            optimizer.zero_grad()
            reconstructed, _ = model(sequences_social, sequences_physiological)
            loss = criterion(reconstructed, sequences)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for sequences, _ in val_dataloader:
                sequences = sequences.to(device)
                sequences_physiological = sequences[:, :, :physiological_input_dim]
                sequences_social = sequences[:, :, physiological_input_dim:]

                reconstructed, _ = model(sequences_social, sequences_physiological)
                loss = criterion(reconstructed, sequences)
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training...")
            model.load_state_dict(torch.load("best_model.pth"))
            break
    return model

def extract_embeddings(model, dataloader_emb, device, input_dims):
    physiological_input_dim, _ = input_dims
    model.eval()
    embeddings, metadata_loader = [], []

    with torch.no_grad():
        for sequences, meta in dataloader_emb:
            sequences = sequences.to(device)
            sequences_physiological = sequences[:, :, :physiological_input_dim]
            sequences_social = sequences[:, :, physiological_input_dim:]
            _, embedding = model(sequences_social, sequences_physiological)
            metadata_loader.append(meta[0])
            embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings), metadata_loader


def perform_clustering(embeddings, n_clusters=12):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embeddings)
    gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit_predict(embeddings)
    return kmeans, gmm


def save_results(model_name, description, metadata_loader, embeddings, kmeans_clusters, gmm_clusters):
    df_clusters = pd.DataFrame(metadata_loader, columns=['user', 'day', 'window'])
    df_clusters['kmeans_cluster'] = kmeans_clusters
    df_clusters['gmm_cluster'] = gmm_clusters
    df_clusters.to_csv(f"results_autoencoder_fusion/csv/all_users_combined_clusters_{model_name}_{description}.csv", index=False)

    embedding_data = {
        "embeddings": embeddings,
        "cluster_labels": {
            "kmeans": kmeans_clusters,
            "gmm": gmm_clusters
        },
        "metadata": metadata_loader
    }
    with open(f"results_autoencoder_fusion/pkl/embeddings_with_clusters_{model_name}_{description}.pkl", "wb") as f:
        pickle.dump(embedding_data, f)

def print_args(args):
    print("Configuration Parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def main():
    args = get_args()
    print_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader_train, dataloader_val, input_dim, dataloader_emb, max_seq_length = get_data_loader(
        args.interval_length, args.overlap, args.scaler, multimodal_autoencoder=True
    )
    physiological_input_dim = len(selected_features_pearson)
    social_input_dim = len(social_columns)

    setup_directories()

    models = ['cnn', 'lstm', 'rnn', 'transformer']
    fusion_types = ['early', 'mid', 'late']

    for model in models:
        for fusion in fusion_types:
            print(f"Training {model} fusion type {fusion}")
            autoencoder = initialize_model(
                model, fusion, (social_input_dim, physiological_input_dim), args.hidden_dim, args.embedding_dim,
                max_seq_length, device, args.intermediate_size, args.num_hidden_layers, args.num_attention_heads
            )
            criterion = nn.MSELoss()
            optimizer = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
            trained_model = train_model(autoencoder, dataloader_train, dataloader_val, criterion, optimizer, args.num_epochs, device, (physiological_input_dim, social_input_dim))
            embeddings, metadata_loader = extract_embeddings(trained_model, dataloader_emb, device, (physiological_input_dim, social_input_dim))
            kmeans_clusters, gmm_clusters = perform_clustering(embeddings)
            save_results(model, fusion, metadata_loader, embeddings, kmeans_clusters, gmm_clusters)


if __name__ == "__main__":
    main()
