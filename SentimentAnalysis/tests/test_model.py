import pytest
import torch
import torch.nn as nn

from src.model import SentimentLSTM

# Define standard hyperparameters for testing
VOCAB_SIZE = 100
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 1 # For sentiment logit
N_LAYERS = 2
DROPOUT_RATE = 0.3
PAD_IDX = 0

@pytest.fixture(scope="module")
def model_unidirectional() -> SentimentLSTM:
    """Provides a non-bidirectional SentimentLSTM model instance."""
    return SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=False,
        dropout_rate=DROPOUT_RATE,
        pad_idx=PAD_IDX,
    )

@pytest.fixture(scope="module")
def model_bidirectional() -> SentimentLSTM:
    """Provides a bidirectional SentimentLSTM model instance."""
    return SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=True,
        dropout_rate=DROPOUT_RATE,
        pad_idx=PAD_IDX,
    )

def test_model_instantiation(model_unidirectional: SentimentLSTM, model_bidirectional: SentimentLSTM):
    """Test if the models are instantiated correctly and have the right components."""
    assert isinstance(model_unidirectional, nn.Module)
    assert isinstance(model_unidirectional.embedding, nn.Embedding)
    assert isinstance(model_unidirectional.lstm, nn.LSTM)
    assert isinstance(model_unidirectional.fc, nn.Linear)
    assert isinstance(model_unidirectional.dropout, nn.Dropout)

    assert isinstance(model_bidirectional, nn.Module)
    assert isinstance(model_bidirectional.embedding, nn.Embedding)
    assert isinstance(model_bidirectional.lstm, nn.LSTM)
    assert isinstance(model_bidirectional.fc, nn.Linear)
    assert isinstance(model_bidirectional.dropout, nn.Dropout)

def test_embedding_pad_idx(model_unidirectional: SentimentLSTM):
    """Test if the padding_idx is correctly set in the embedding layer."""
    assert model_unidirectional.embedding.padding_idx == PAD_IDX

def test_lstm_bidirectional_param(model_unidirectional: SentimentLSTM, model_bidirectional: SentimentLSTM):
    """Test if the bidirectional flag is correctly set in the LSTM layer."""
    assert model_unidirectional.lstm.bidirectional is False
    assert model_bidirectional.lstm.bidirectional is True

def test_fc_layer_input_dim(model_unidirectional: SentimentLSTM, model_bidirectional: SentimentLSTM):
    """Test if the fully connected layer's input dimension is correct based on bidirectionality."""
    # Unidirectional: fc input should be hidden_dim
    assert model_unidirectional.fc.in_features == HIDDEN_DIM
    # Bidirectional: fc input should be hidden_dim * 2
    assert model_bidirectional.fc.in_features == HIDDEN_DIM * 2

def test_dropout_rate(model_unidirectional: SentimentLSTM):
    """Test if the dropout rate is correctly set."""
    # For the main dropout layer applied to embeddings and hidden states
    assert model_unidirectional.dropout.p == DROPOUT_RATE
    # For LSTM internal dropout (if n_layers > 1)
    if N_LAYERS > 1:
        assert model_unidirectional.lstm.dropout == DROPOUT_RATE
    else:
        assert model_unidirectional.lstm.dropout == 0

@pytest.mark.parametrize("bidirectional_model_fixture_name", ["model_unidirectional", "model_bidirectional"])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("seq_len", [10, 50])
def test_forward_pass_output_shape(bidirectional_model_fixture_name: str, batch_size: int, seq_len: int, request):
    """Test the forward pass and output shape for both model types."""
    model: SentimentLSTM = request.getfixturevalue(bidirectional_model_fixture_name)
    model.eval() # Set to evaluation mode

    # Create dummy input tensor (batch_size, seq_len)
    dummy_input = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), dtype=torch.long)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (batch_size, OUTPUT_DIM)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_on_cuda(model_bidirectional: SentimentLSTM):
    """Test if the model can be moved to and run on CUDA (if available)."""
    device = torch.device("cuda")
    model_on_cuda = model_bidirectional.to(device)
    model_on_cuda.eval()

    dummy_input = torch.randint(0, VOCAB_SIZE, (4, 20), dtype=torch.long).to(device)
    
    with torch.no_grad():
        try:
            output = model_on_cuda(dummy_input)
            assert output.device.type == "cuda"
            assert output.shape == (4, OUTPUT_DIM)
        except RuntimeError as e:
            pytest.fail(f"Model failed to run on CUDA: {e}")

def test_forward_pass_with_padding(model_bidirectional: SentimentLSTM):
    """Test forward pass with sequences that would include padding.
    This mainly ensures it runs without error; precise output checking with padding is complex.
    """
    model_bidirectional.eval()
    batch_size = 2
    seq_len = 10 # Assume max_len for padding

    # Input includes PAD_IDX
    input_tensor = torch.tensor([
        [10, 20, 30, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX],
        [15, 25, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]
    ], dtype=torch.long)
    
    if PAD_IDX >= VOCAB_SIZE:
        pytest.skip("PAD_IDX is out of VOCAB_SIZE for this dummy tensor construction")

    with torch.no_grad():
        output = model_bidirectional(input_tensor)
    
    assert output.shape == (batch_size, OUTPUT_DIM) 