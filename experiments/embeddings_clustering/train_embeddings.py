import argparse
import os
import pickle
import warnings
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dataset.load_data_utils import get_data_loader_noCV
from models.embeddings.embeddings_models import RNNAutoencoder, LSTM_Autoencoder, TransformerAutoencoder, CNN_Autoencoder, BERTAutoencoder, apply_mask

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Clustering parameters')
    parser.add_argument('--interval_length', type=int, default=8, help='Choice of window length')
    parser.add_argument('--overlap', type=int, default=4, help='Choice of overlap for time windows')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden Dimension (Default: 1024)')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding Dimension (Default: 128)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of Training Epochs (Default: 100)')
    parser.add_argument('--scaler', type=str, default='Standard', choices=['MinMax', 'Standard'], help='Choice of scaler: MinMax for MinMaxScaler, Standard for StandardScaler')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of hidden layers (Default: 1)')
    parser.add_argument('--num_attention_heads', type=int, default=1, help='Number of attention heads (Default: 1)')
    parser.add_argument('--intermediate_size', type=int, default=512, help='Intermediate size for feed-forward layers (Default: 256)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer to use (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate (default: 1e-3)')
    parser.add_argument('--use_simple_decoder', action='store_true', default=False, help='Use simple decoder (Default: False)')
    parser.add_argument('--use_dropout', action='store_true', default=False, help='Use dropout (Default: False)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate (Default: 0.5)')
    parser.add_argument('--add_noise', action='store_true', default=False, help='Add noise (Default: False)')
    parser.add_argument('--noise_factor', type=float, default=0.3, help='Noise factor (Default: 0.3)')
    return parser.parse_args()

def train_autoencoder(model, dataloader, device, num_epochs, lr, optimizer):
    criterion_recon = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer == 'adam' else optim.SGD(model.parameters(), lr=lr)

    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epochs "):
        epoch_loss = 0
        for sequences, _ in dataloader:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            reconstructed, embeddings = model(sequences)
            loss = criterion_recon(reconstructed, sequences)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1)  == num_epochs:
            print(f"Final Loss: {epoch_loss / len(dataloader)}")
        # elif (epoch + 1) % 10 == 0:
        #     print(f"Loss: {epoch_loss / len(dataloader)}")

    return model

def masked_ts_modeling_training(model, dataloader, device, num_epochs, lr, optimizer):
    criterion_recon = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer == 'adam' else optim.SGD(model.parameters(), lr=lr)

    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epochs "):
        epoch_loss = 0
        for sequences, _ in dataloader:
            sequences = sequences.to(device)
            masked_sequences, mask = apply_mask(sequences)
            masked_sequences = masked_sequences.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            reconstructed, embeddings = model(masked_sequences)

            # Calculate loss only for masked parts
            loss = criterion_recon(reconstructed, sequences)
            loss = (loss * mask).sum() / mask.sum()  # Average loss over masked elements

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1)  == num_epochs:
            print(f"Final Loss: {epoch_loss / len(dataloader)}")
        # elif (epoch + 1) % 10  == 0:
        #     print(f"Loss: {epoch_loss / len(dataloader)}")

    return model


def extract_and_save_embeddings(model, dataloader, device, model_name, suffix = ''):
    model.eval()
    embeddings = []
    metadata_loader = []
    dataloader_emb = DataLoader(dataloader.dataset, batch_size=1, shuffle=True, collate_fn=dataloader.collate_fn)
    os.makedirs("results_embeddings/pkl/", exist_ok=True)
    with torch.no_grad():
        for sequences, meta in dataloader_emb:
            sequences = sequences.to(device)
            _, embedding = model(sequences)
            metadata_loader.append(meta[0])
            embeddings.append(embedding.cpu().numpy())
    embeddings = np.vstack(embeddings)
    embedding_data = {"embeddings": embeddings, "metadata": metadata_loader}
    with open(f"results_embeddings/pkl/embedding_data_{model_name}{suffix}.pkl", "wb") as f:
        pickle.dump(embedding_data, f)


def save_model(model, model_name):
    os.makedirs("results_embeddings/models", exist_ok=True)
    model_path = f"results_embeddings/models/{model_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def print_args(args):
    print("Configuration Parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def main():
    args = get_args()
    print_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, input_dim = get_data_loader_noCV(args.interval_length, args.overlap, args.scaler)

    os.makedirs("results_embeddings/pkl", exist_ok=True)

    models = {
        "lstm": LSTM_Autoencoder(input_dim, args.hidden_dim, args.embedding_dim, use_simple_decoder= args.use_simple_decoder, use_dropout=args.use_dropout, dropout_rate=args.dropout_rate, add_noise=args.add_noise, noise_factor=args.noise_factor).to(device),
        "cnn": CNN_Autoencoder(input_dim, args.hidden_dim, args.embedding_dim, use_simple_decoder= args.use_simple_decoder, use_dropout=args.use_dropout, dropout_rate=args.dropout_rate, add_noise=args.add_noise, noise_factor=args.noise_factor).to(device),
        "rnn": RNNAutoencoder(input_dim, args.hidden_dim, args.embedding_dim, use_simple_decoder= args.use_simple_decoder, use_dropout=args.use_dropout, dropout_rate=args.dropout_rate, add_noise=args.add_noise, noise_factor=args.noise_factor).to(device),
        "tsf": TransformerAutoencoder(input_dim, args.embedding_dim, args.num_hidden_layers, args.num_attention_heads, args.intermediate_size, use_simple_decoder= args.use_simple_decoder, use_dropout=args.use_dropout, dropout_rate=args.dropout_rate, add_noise=args.add_noise, noise_factor=args.noise_factor).to(device),
        "bert": BERTAutoencoder(input_dim, args.embedding_dim, args.num_hidden_layers, args.num_attention_heads, args.intermediate_size, use_simple_decoder= args.use_simple_decoder, use_dropout=args.use_dropout, dropout_rate=args.dropout_rate, add_noise=args.add_noise, noise_factor=args.noise_factor).to(device)
    }
    for model_name, model in models.items():
        if model_name == "bert":
            model = masked_ts_modeling_training(model, dataloader, device, args.num_epochs, args.lr, args.optimizer)
        else:
            model = train_autoencoder(model, dataloader, device, args.num_epochs, args.lr, args.optimizer)

        extract_and_save_embeddings(model, dataloader, device, model_name)
        save_model(model, model_name)

    print("All models trained, encoders saved, and results saved.")

if __name__ == "__main__":
    main()
