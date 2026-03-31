import torch
import torch.nn as nn
from typing import Tuple

class SentimentLSTM(nn.Module):
    """LSTM model for sentiment analysis.

    The model consists of an embedding layer, an LSTM layer, a dropout layer,
    and a fully connected layer. It outputs raw logits for sentiment classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,  # Should typically be 1 for a sentiment score (logit)
        n_layers: int,
        bidirectional: bool,
        dropout_rate: float,
        pad_idx: int,
    ) -> None:
        """Initializes the SentimentLSTM model.

        Args:
            vocab_size: The size of the vocabulary (number of unique tokens).
            embedding_dim: The dimensionality of the word embeddings.
            hidden_dim: The dimensionality of the LSTM hidden states.
            output_dim: The dimensionality of the output layer. For sentiment score,
                        this is typically 1.
            n_layers: The number of recurrent layers in the LSTM.
            bidirectional: If True, a bidirectional LSTM is used.
            dropout_rate: The dropout probability.
            pad_idx: The index of the padding token in the vocabulary, which will be
                     ignored by the embedding layer.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if n_layers > 1 else 0, # Dropout only between LSTM layers if n_layers > 1
            batch_first=True  # Input/output tensors are (batch, seq, feature)
        )

        self.dropout = nn.Dropout(dropout_rate)

        # If bidirectional, the linear layer input size is hidden_dim * 2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

    def forward(self, text_sequences: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            text_sequences: A batch of input sequences, with shape
                            (batch_size, sequence_length).

        Returns:
            A tensor of output logits, with shape (batch_size, output_dim).
            For sentiment score (output_dim=1), this will be (batch_size, 1),
            which can be squeezed to (batch_size).
        """
        # text_sequences shape: (batch_size, seq_len)
        
        embedded = self.dropout(self.embedding(text_sequences))
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # packed_output, (hidden, cell) = self.lstm(embedded)
        # If using packed sequences (recommended for variable length sequences to avoid processing padding):
        #   lengths = torch.tensor([len(seq) for seq in text_sequences]) # This needs to be actual lengths BEFORE padding
        #   packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        #   packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #   lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Else (simpler, but processes padding tokens through LSTM):
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output shape: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden shape: (n_layers * num_directions, batch_size, hidden_dim)
        # cell shape: (n_layers * num_directions, batch_size, hidden_dim)

        # We are interested in the hidden state of the last time step for classification.
        # If bidirectional, concatenate the final forward and backward hidden states.
        if self.bidirectional:
            # hidden is (n_layers * 2, batch_size, hidden_dim)
            # Concatenate the hidden states of the last layer: 
            # hidden[-2,:,:] is the last forward RNN hidden state
            # hidden[-1,:,:] is the last backward RNN hidden state
            final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # hidden is (n_layers, batch_size, hidden_dim)
            # Take the hidden state of the last layer
            final_hidden = hidden[-1, :, :]
        # final_hidden shape: (batch_size, hidden_dim * num_directions)
        
        # Apply dropout to the concatenated/final hidden states before the FC layer
        dropped_hidden = self.dropout(final_hidden)

        # Pass through the fully connected layer
        output_logits = self.fc(dropped_hidden)
        # output_logits shape: (batch_size, output_dim)

        return output_logits


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing SentimentLSTM model instantiation...")

    # Hyperparameters for a sample model
    VOCAB_SIZE = 5000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1  # For a single sentiment score (logit)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT_RATE = 0.5
    PAD_IDX = 0 # Assuming pad token is at index 0

    # Instantiate the model
    model = SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout_rate=DROPOUT_RATE,
        pad_idx=PAD_IDX,
    )

    logger.info(f"Model instantiated successfully:\n{model}")

    # Test with a dummy batch of input data
    BATCH_SIZE = 4
    SEQ_LEN = 50 
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)) # (batch_size, seq_len)
    logger.info(f"\nTesting forward pass with dummy input of shape: {dummy_input.shape}")
    
    try:
        model.eval() # Set model to evaluation mode for inference
        with torch.no_grad(): # Disable gradient calculations
            predictions = model(dummy_input)
        logger.info(f"Dummy input predictions (logits) shape: {predictions.shape}")
        if predictions.shape == (BATCH_SIZE, OUTPUT_DIM):
            logger.info("Forward pass successful and output shape is correct.")
        else:
            logger.error(
                f"Forward pass output shape is incorrect. Expected: {(BATCH_SIZE, OUTPUT_DIM)}, Got: {predictions.shape}"
            )
        # For sentiment score, apply sigmoid
        scores = torch.sigmoid(predictions)
        logger.info(f"Dummy scores (after sigmoid) sample: {scores.squeeze()[:2].tolist()}...")

    except Exception as e:
        logger.error(f"Error during model forward pass test: {e}", exc_info=True)

    logger.info("\nSentimentLSTM model test finished.") 