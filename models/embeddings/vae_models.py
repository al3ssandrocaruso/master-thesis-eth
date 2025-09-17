#from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_mask(x, mask_ratio=0.3):
    seq_length, batch_size, hidden_dim = x.size()
    masked_x = x.clone()
    mask = torch.zeros(seq_length, batch_size, dtype=torch.bool)

    for b in range(batch_size):
        # Create mask for current batch element
        num_masked = int(mask_ratio * seq_length)
        mask_indices = torch.randperm(seq_length)[:num_masked]
        mask[mask_indices, b] = True

    masked_x[mask] = 0
    return masked_x, mask

def add_noise(inputs, noise_factor=0.3):
    noisy_inputs = inputs + noise_factor * torch.randn_like(inputs)
    noisy_inputs = torch.clip(noisy_inputs, 0., 1.)
    return noisy_inputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Loss function for VAE
def vae_loss(reconstructed, x, mu, log_var):
    """
    Compute the VAE loss as the sum of reconstruction loss and KL divergence.

    Args:
        reconstructed: Reconstructed input.
        x: Original input.
        mu: Mean of the latent space.
        log_var: Log variance of the latent space.

    Returns:
        Total loss, reconstruction loss, and KL divergence.
    """
    recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + kld_loss
    return total_loss, recon_loss, kld_loss

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, use_simple_decoder=False, use_dropout=False,
                 dropout_rate=0.5):
        super(LSTM_VAE, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dim, embedding_dim)
        self.fc_logvar = nn.Linear(hidden_dim,embedding_dim)

        self.use_simple_decoder = use_simple_decoder
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        # Decoder setup
        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode input
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]  # Use the last layer's hidden state
        
        # Compute latent variables
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Reparameterization trick
        z = reparameterize(mu, logvar)
        if self.use_dropout:
            z = self.dropout(z)
        
        # Decode latent variable
        if self.use_simple_decoder:
            reconstructed = self.decoder(z).unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            z_repeated = z.unsqueeze(1).repeat(1, x.size(1), 1)
            decoder_output, _ = self.decoder(z_repeated)
            reconstructed = self.output(decoder_output)
        
        return reconstructed, x, mu, logvar, z

    def get_encoder(self):
        return self.encoder

class RNN_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, use_simple_decoder=False, use_dropout=False, dropout_rate=0.5, add_noise=False, noise_factor=0.3):
        super(RNN_VAE, self).__init__()
        self.encoder = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, embedding_dim)        # Mean
        self.fc_logvar = nn.Linear(hidden_dim, embedding_dim)   # Log-variance
        self.use_simple_decoder = use_simple_decoder
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            self.decoder = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            self.output = nn.Linear(hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        """Apply reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random normal noise
        return mu + eps * std

    def forward(self, x):
        if self.add_noise:
            x_noisy = add_noise(x, self.noise_factor)
        else:
            x_noisy = x

        _, hidden = self.encoder(x_noisy)
        mu = self.fc_mu(hidden[-1])
        logvar = self.fc_logvar(hidden[-1])
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        if self.use_dropout:
            z = self.dropout(z)

        # Decode
        if self.use_simple_decoder:
            reconstructed = self.decoder(z)
            reconstructed = reconstructed.unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            repeated_z = z.unsqueeze(1).repeat(1, x.size(1), 1)
            decoder_output, _ = self.decoder(repeated_z)
            reconstructed = self.output(decoder_output)

        return reconstructed, x, mu, logvar, z

    def get_encoder(self):
        return self.encoder


class Transformer_VAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_hidden_layers, num_attention_heads, intermediate_size,
                 ti_mae=False, use_simple_decoder=False, use_dropout=False, dropout_rate=0.3, add_noise=False,
                 noise_factor=0.3):
        super(Transformer_VAE, self).__init__()

        self.ti_mae = ti_mae
        self.use_simple_decoder = use_simple_decoder
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        self.noise_factor = noise_factor

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim)

        # Encoder using Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=intermediate_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

        # Linear layers for dimensionality adjustment
        self.input_to_hidden = nn.Linear(input_dim, embedding_dim)

        # Latent space for VAE
        self.fc_mu = nn.Linear(embedding_dim, embedding_dim)
        self.fc_log_var = nn.Linear(embedding_dim, embedding_dim)

        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            # Decoder using Transformer
            decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=intermediate_size)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_hidden_layers)

            # Output layer to reconstruct the input
            self.output = nn.Linear(embedding_dim, input_dim)

        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample z ~ N(mu, sigma^2)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.add_noise:
            x_noisy = add_noise(x, self.noise_factor)
        else:
            x_noisy = x

        # Adjust input dimensions to hidden dimensions
        x_hidden = self.input_to_hidden(x_noisy)

        # Add positional encoding
        x_hidden = self.positional_encoding(x_hidden.permute(1, 0, 2))  # (seq_length, batch_size, hidden_dim)
        if self.ti_mae:
            # Apply masking
            masked_x_hidden, mask = apply_mask(x_hidden)
        else:
            masked_x_hidden = x_hidden

        # Encoding
        encoder_output = self.encoder(masked_x_hidden)  # (seq_length, batch_size, hidden_dim)

        # Get mean and log variance for latent space
        mu = self.fc_mu(encoder_output.mean(dim=0))  # (batch_size, hidden_dim)
        log_var = self.fc_log_var(encoder_output.mean(dim=0))  # (batch_size, hidden_dim)

        # Latent vector
        z = self.reparameterize(mu, log_var)

        if self.use_simple_decoder:
            reconstructed = self.decoder(z)
            reconstructed = reconstructed.unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            # Prepare repeated latent vector for the decoder input
            repeated_z = z.unsqueeze(1).repeat(1, x.size(1), 1).permute(1, 0, 2)  # (seq_length, batch_size, hidden_dim)

            # Add positional encoding to the decoder input
            repeated_z = self.positional_encoding(repeated_z)

            # Create a mask for the decoder to ignore masked positions
            if self.ti_mae:
                # Invert the mask to attend only non-masked tokens
                memory_key_padding_mask = ~mask.permute(1, 0).to(x.device)
            else:
                memory_key_padding_mask = None

            # Decoding
            decoder_output = self.decoder(
                tgt=repeated_z,
                memory=encoder_output,
                memory_key_padding_mask=memory_key_padding_mask
            )  # (seq_length, batch_size, hidden_dim)

            reconstructed_hidden = decoder_output.permute(1, 0, 2)  # (batch_size, seq_length, hidden_dim)

            # Adjust hidden dimensions back to input dimensions
            reconstructed = self.output(reconstructed_hidden)  # (batch_size, seq_length, input_dim)

        return reconstructed, x, mu, log_var, z
    def get_encoder(self):
        return nn.ModuleList([self.encoder, self.input_to_hidden])


class CNN_VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        kernel_size=3,
        use_simple_decoder=False,
        use_dropout=False,
        dropout_rate=0.5,
        add_noise=False,
        noise_factor=0.3
    ):
        super(CNN_VAE, self).__init__()

        self.use_simple_decoder = use_simple_decoder
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        self.noise_factor = noise_factor

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(embedding_dim, embedding_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(embedding_dim, embedding_dim)  # Log variance of latent distribution

        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(
                    embedding_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose1d(
                    hidden_dim, input_dim, kernel_size=kernel_size, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid()
            )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Sample from standard normal distribution
        return mu + eps * std

    def forward(self, x):
        if self.add_noise:
            x_noisy = add_noise(x, self.noise_factor)
        else:
            x_noisy = x

        # Permute to (batch_size, input_dim, seq_length) for Conv1d
        x_noisy = x_noisy.permute(0, 2, 1)

        # Encoding
        encoded = self.encoder(x_noisy)

        # Get the embedding of shape [batch_size, embedding_dim]
        batch_size, embedding_dim, seq_length = encoded.shape
        embedding = torch.mean(encoded, dim=2)  # Take the mean across the sequence length

        # Compute latent distribution parameters
        mu = self.fc_mu(embedding)
        logvar = self.fc_logvar(embedding)

        # Reparameterize to get latent vector z
        z = self.reparameterize(mu, logvar)

        if self.use_dropout:
            z = self.dropout(z)

        if self.use_simple_decoder:
            reconstructed = self.decoder(z)
            reconstructed = reconstructed.unsqueeze(2).repeat(1, 1, x.size(2))
        else:
            # Reshape z back to [batch_size, embedding_dim, 1] for decoding
            z_for_decoding = z.unsqueeze(2).repeat(1, 1, seq_length)

            # Decoding
            decoded = self.decoder(z_for_decoding)

            # Permute back to (batch_size, seq_length, input_dim)
            reconstructed = decoded.permute(0, 2, 1)

        return reconstructed, x, mu, logvar, z

    def get_encoder(self):
        return self.encoder

