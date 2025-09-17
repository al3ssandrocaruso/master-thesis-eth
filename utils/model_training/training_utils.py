import torch
import os
from models.multimodal.multimodal_embeddings_models import GeneralizedLateFusionAutoencoder, GeneralizedMidFusionAutoencoder,GeneralizedEarlyFusionAutoencoder
from models.embeddings.vae_models import vae_loss
from tqdm import tqdm
from config.config import BASE_DIR
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Multimodal Embedding Training and Embedding Extraction')
    parser.add_argument('--interval_length', type=int, default=24, help='Interval length for data windows')
    parser.add_argument('--overlap', type=int, default=12, help='Overlap between data windows')
    parser.add_argument('--scaler', type=str, default='Standard', choices=['MinMax', 'Standard'], help='Scaler for preprocessing data')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of hidden layers (Default: 1)')
    parser.add_argument('--num_attention_heads', type=int, default=1, help='Number of attention heads (Default: 1)')
    parser.add_argument('--intermediate_size', type=int, default=256, help='Intermediate size for feed-forward layers (Default: 512)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument("--model", type=str, help="Model to run")

    return parser.parse_args()

def print_args(args):
    print("Configuration Parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, save_path="./best_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")  # Best loss so far
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            # Update the best loss and save the model if validation loss improves
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        else:
            # Increment counter if validation loss does not improve
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss improves."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.save_path)


def setup_directories():
    paths = [
        f"{BASE_DIR}/results_autoencoder_fusion/csv",
        f"{BASE_DIR}/results_autoencoder_fusion/img",
        f"{BASE_DIR}/results_autoencoder_fusion/pkl",
        f"{BASE_DIR}/results_autoencoder_fusion_clf/pkl",
        f"{BASE_DIR}/intermediate_data/pkl",
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

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

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, input_dims, vae):
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
            if vae:
                reconstructed, x, mu, logvar, z = model(sequences_social, sequences_physiological)
                loss,_,_ = vae_loss(reconstructed,x,mu,logvar)
            else:
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
                if vae:
                    reconstructed, x, mu, logvar, z = model(sequences_social, sequences_physiological)
                    loss,_,_ = vae_loss(reconstructed,x,mu,logvar)
                else:
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
            model.load_state_dict(torch.load("./best_model.pth"))
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
