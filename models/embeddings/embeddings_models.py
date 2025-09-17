#from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
import math

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, use_simple_decoder=False, use_dropout=False,
                 dropout_rate=0.5, add_noise=False, noise_factor=0.3):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.embedding = nn.Linear(hidden_dim, embedding_dim)
        self.use_simple_decoder = use_simple_decoder
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if self.add_noise:
            x_noisy = add_noise(x, self.noise_factor)
        else:
            x_noisy = x

        _, (hidden, _) = self.encoder(x_noisy)
        embedding = self.embedding(hidden[-1])
        if self.use_dropout:
            embedding = self.dropout(embedding)

        if self.use_simple_decoder:
            reconstructed = self.decoder(embedding)
            reconstructed = reconstructed.unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            repeated_embedding = embedding.unsqueeze(1).repeat(1, x.size(1), 1)
            decoder_output, _ = self.decoder(repeated_embedding)
            reconstructed = self.output(decoder_output)

        return reconstructed, embedding

    def get_encoder(self):
        return self.encoder

class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, use_simple_decoder=False, use_dropout=False, dropout_rate=0.5, add_noise=False, noise_factor=0.3):
        super(RNNAutoencoder, self).__init__()
        self.encoder = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.embedding = nn.Linear(hidden_dim, embedding_dim)
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

    def forward(self, x):
        if self.add_noise:
            x_noisy = add_noise(x, self.noise_factor)
        else:
            x_noisy = x

        _, hidden = self.encoder(x_noisy)
        embedding = self.embedding(hidden[-1])
        if self.use_dropout:
            embedding = self.dropout(embedding)

        if self.use_simple_decoder:
            reconstructed = self.decoder(embedding)
            reconstructed = reconstructed.unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            repeated_embedding = embedding.unsqueeze(1).repeat(1, x.size(1), 1)
            decoder_output, _ = self.decoder(repeated_embedding)
            reconstructed = self.output(decoder_output)

        return reconstructed, embedding

    def get_encoder(self):
        return self.encoder

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

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_hidden_layers, num_attention_heads, intermediate_size,
                 ti_mae=False, use_simple_decoder=False, use_dropout=False, dropout_rate=0.3, add_noise=False,
                 noise_factor=0.3):
        super(TransformerAutoencoder, self).__init__()

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
        self.hidden_to_input = nn.Linear(embedding_dim, input_dim)

        # Linear layer to get the embedding from the last hidden state
        self.embedding = nn.Linear(embedding_dim, embedding_dim)
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            # Decoder using Transformer's layers
            decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=intermediate_size)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_hidden_layers)

            # Output layer to reconstruct the input
            self.output = nn.Linear(embedding_dim, input_dim)

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

        # Get the embedding from the last hidden state
        embedding = self.embedding(encoder_output.mean(dim=0))  # (batch_size, hidden_dim)
        if self.use_dropout:
            embedding = self.dropout(embedding)

        if self.use_simple_decoder:
            reconstructed = self.decoder(embedding)
            reconstructed = reconstructed.unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            # Prepare repeated embedding for the decoder input
            repeated_embedding = embedding.unsqueeze(1).repeat(1, x.size(1), 1).permute(1, 0, 2)  # (seq_length, batch_size, hidden_dim)

            # Add positional encoding to the decoder input
            repeated_embedding = self.positional_encoding(repeated_embedding)

            # Create a mask for the decoder to ignore masked positions
            if self.ti_mae:
                # Invert the mask to attend only non-masked tokens
                memory_key_padding_mask = ~mask.permute(1, 0).to(x.device)
            else:
                memory_key_padding_mask = None

            # Decoding
            decoder_output = self.decoder(
                tgt=repeated_embedding,
                memory=encoder_output,
                memory_key_padding_mask=memory_key_padding_mask
            )  # (seq_length, batch_size, hidden_dim)

            reconstructed_hidden = decoder_output.permute(1, 0, 2)  # (batch_size, seq_length, hidden_dim)

            # Adjust hidden dimensions back to input dimensions
            reconstructed = self.hidden_to_input(reconstructed_hidden)  # (batch_size, seq_length, input_dim)

        return reconstructed, embedding

    def get_encoder(self):
        return nn.ModuleList([self.encoder, self.input_to_hidden])

class CNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, kernel_size=3, use_simple_decoder=False, use_dropout=False, dropout_rate=0.5, add_noise=False, noise_factor=0.3):
        super(CNN_Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU()
        )
        self.use_simple_decoder = use_simple_decoder
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1,output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=kernel_size, stride=2, padding=1,output_padding=1),
                nn.Sigmoid()
            )

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
        if self.use_dropout:
            embedding = self.dropout(embedding)

        if self.use_simple_decoder:
            reconstructed = self.decoder(embedding)
            reconstructed = reconstructed.unsqueeze(2).repeat(1, 1, x.size(2))
        else:
            # Reshape embedding back to [batch_size, embedding_dim, 1] for decoding
            encoded_for_decoding = embedding.unsqueeze(2).repeat(1, 1, seq_length)

            # Decoding
            decoded = self.decoder(encoded_for_decoding)

            # Permute back to (batch_size, seq_length, input_dim)
            reconstructed = decoded.permute(0, 2, 1)

        return reconstructed, embedding

    def get_encoder(self):
        return self.encoder

class BERTAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_hidden_layers, num_attention_heads, intermediate_size, use_simple_decoder=False, use_dropout=False, dropout_rate=0.5, add_noise=False, noise_factor=0.3):
        super(BERTAutoencoder, self).__init__()

        # BERT configuration
        self.config = BertConfig(
            hidden_size=embedding_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act="gelu",
            max_position_embeddings=512,
            vocab_size=30522,  # This won't be used, but is a required argument
            type_vocab_size=2
        )

        # Encoder using BERT
        self.encoder = BertModel(self.config)

        # Linear layers for dimensionality adjustment
        self.input_to_hidden = nn.Linear(input_dim, embedding_dim)
        self.hidden_to_input = nn.Linear(embedding_dim, input_dim)

        # Linear layer to get the embedding from the last hidden state
        self.embedding = nn.Linear(embedding_dim, embedding_dim)
        self.use_simple_decoder = use_simple_decoder
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        if use_simple_decoder:
            self.decoder = nn.Linear(embedding_dim, input_dim)
        else:
            # Decoder using BERT's layers
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=intermediate_size)
            self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_hidden_layers)

            # Output layer to reconstruct the input
            self.output = nn.Linear(embedding_dim, input_dim)

    def forward(self, x):
        if self.add_noise:
            x_noisy = add_noise(x, self.noise_factor)
        else:
            x_noisy = x

        # Adjust input dimensions to hidden dimensions
        x_hidden = self.input_to_hidden(x_noisy)

        # Encoding
        encoder_outputs = self.encoder(inputs_embeds=x_hidden)
        last_hidden_state = encoder_outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
        embedding = self.embedding(last_hidden_state.mean(dim=1))  # (batch_size, hidden_dim)
        if self.use_dropout:
            embedding = self.dropout(embedding)

        if self.use_simple_decoder:
            reconstructed = self.decoder(embedding)
            reconstructed = reconstructed.unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            # Prepare repeated embedding for the decoder input
            repeated_embedding = embedding.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch_size, seq_length, hidden_dim)

            # Decoding
            decoder_output = self.decoder(repeated_embedding.permute(1, 0, 2), last_hidden_state.permute(1, 0, 2))  # (seq_length, batch_size, hidden_dim)
            reconstructed_hidden = decoder_output.permute(1, 0, 2)  # (batch_size, seq_length, hidden_dim)

            # Adjust hidden dimensions back to input dimensions
            reconstructed = self.hidden_to_input(reconstructed_hidden)  # (batch_size, seq_length, input_dim)

        return reconstructed, embedding

    def get_encoder(self):
        return self.encoder

def add_noise(inputs, noise_factor=0.3):
    noisy_inputs = inputs + noise_factor * torch.randn_like(inputs)
    noisy_inputs = torch.clip(noisy_inputs, 0., 1.)
    return noisy_inputs
