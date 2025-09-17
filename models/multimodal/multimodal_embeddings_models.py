import torch
import torch.nn as nn

from models.embeddings.embeddings_models import LSTM_Autoencoder, RNNAutoencoder, CNN_Autoencoder, TransformerAutoencoder, \
    BERTAutoencoder
from models.embeddings.vae_models import LSTM_VAE, CNN_VAE, Transformer_VAE, RNN_VAE


class AutoencoderWrapper(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, embedding_dim, num_hidden_layers=None, num_attention_heads=None, intermediate_size=None):
        super(AutoencoderWrapper, self).__init__()
        if model_type == 'lstm':
            self.model = LSTM_Autoencoder(input_dim, hidden_dim, embedding_dim)
        elif model_type == 'lstm_vae':
            self.model = LSTM_VAE(input_dim, hidden_dim, embedding_dim)
        elif model_type == 'rnn':
            self.model = RNNAutoencoder(input_dim, hidden_dim, embedding_dim)
        elif model_type == 'rnn_vae':
            self.model = RNN_VAE(input_dim, hidden_dim, embedding_dim)
        elif model_type == 'cnn':
            self.model = CNN_Autoencoder(input_dim, hidden_dim, embedding_dim)
        elif model_type == 'cnn_vae':
            self.model = CNN_VAE(input_dim, hidden_dim, embedding_dim)
        elif model_type == 'transformer':
            self.model = TransformerAutoencoder(input_dim, embedding_dim, num_attention_heads, num_hidden_layers, intermediate_size)
        elif model_type == 'transformer_vae':
            self.model = Transformer_VAE(input_dim, embedding_dim, num_attention_heads, num_hidden_layers, intermediate_size)
        elif model_type == 'bert':
            self.model = BERTAutoencoder(input_dim, embedding_dim, num_hidden_layers, num_attention_heads, intermediate_size)
        else:
            raise ValueError("Unsupported model type")

    def forward(self, x):
        return self.model(x)


class GeneralizedLateFusionAutoencoder(nn.Module):
    def __init__(self, social_model_type, physiological_model_type, social_input_dim, physiological_input_dim, hidden_dim, embedding_dim, num_hidden_layers=None, num_attention_heads=None, intermediate_size=None):
        super(GeneralizedLateFusionAutoencoder, self).__init__()
        self.social_autoencoder = AutoencoderWrapper(social_model_type, social_input_dim, hidden_dim, embedding_dim, num_hidden_layers, num_attention_heads, intermediate_size)
        self.physiological_autoencoder = AutoencoderWrapper(physiological_model_type, physiological_input_dim, hidden_dim, embedding_dim, num_hidden_layers, num_attention_heads, intermediate_size)

        # Fusion layer to combine reconstructed outputs
        self.fusion_layer = nn.Linear(social_input_dim + physiological_input_dim, social_input_dim + physiological_input_dim)

    def forward(self, social_data, physiological_data):
        social_reconstructed, social_embedding = self.social_autoencoder(social_data)
        physiological_reconstructed, physiological_embedding = self.physiological_autoencoder(physiological_data)

        combined_reconstructed = torch.cat((social_reconstructed, physiological_reconstructed), dim=-1)
        fused_output = self.fusion_layer(combined_reconstructed)

        return fused_output, torch.cat((social_embedding, physiological_embedding), dim=-1)


class GeneralizedMidFusionAutoencoder(nn.Module):
    def __init__(self, social_model_type, physiological_model_type, social_input_dim, physiological_input_dim, seq_len, hidden_dim, embedding_dim, num_hidden_layers=None, num_attention_heads=None, intermediate_size=None):
        super(GeneralizedMidFusionAutoencoder, self).__init__()
        self.social_autoencoder = AutoencoderWrapper(social_model_type, social_input_dim, hidden_dim, embedding_dim,
                                                     num_hidden_layers, num_attention_heads, intermediate_size)
        self.physiological_autoencoder = AutoencoderWrapper(physiological_model_type, physiological_input_dim,
                                                            hidden_dim, embedding_dim, num_hidden_layers,
                                                            num_attention_heads, intermediate_size)

        self.seq_len = seq_len

        # Fusion layer to combine embeddings
        self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)

        # Enhanced decoder to output [batch_size, seq_len, input_dim]
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * (social_input_dim + physiological_input_dim)),
            nn.ReLU()
        )
        self.output_dim = social_input_dim + physiological_input_dim

    def forward(self, social_data, physiological_data):
        social_reconstructed, social_embedding = self.social_autoencoder(social_data)
        physiological_reconstructed, physiological_embedding = self.physiological_autoencoder(physiological_data)

        combined_embedding = torch.cat((social_embedding, physiological_embedding), dim=-1)
        fused_embedding = self.fusion_layer(combined_embedding)
        reconstructed_output = self.decoder(fused_embedding)

        # Reshape to [batch_size, seq_len, input_dim]
        reconstructed_output = reconstructed_output.view(-1, self.seq_len, self.output_dim)

        return reconstructed_output, fused_embedding


class GeneralizedEarlyFusionAutoencoder(nn.Module):
    def __init__(self, model_type, social_input_dim, physiological_input_dim, hidden_dim, embedding_dim,
                 num_hidden_layers=None, num_attention_heads=None, intermediate_size=None):
        super(GeneralizedEarlyFusionAutoencoder, self).__init__()

        self.input_dim = social_input_dim + physiological_input_dim
        self.model = AutoencoderWrapper(model_type, self.input_dim, hidden_dim, embedding_dim, num_hidden_layers,
                                        num_attention_heads, intermediate_size)

    def forward(self, social_data, physiological_data):
        combined_input = torch.cat((social_data, physiological_data), dim=-1)

        try:
            reconstructed_output, embedding = self.model(combined_input)
            return reconstructed_output, embedding
        except:
            reconstructed_output,x,mu,logvar,embedding = self.model(combined_input)
            return reconstructed_output,x,mu,logvar,embedding