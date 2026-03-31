import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch
from typing import List

from src.trainer import ModelTrainer, convert_score_to_categorical

# --- Helper Mock Classes ---
class MockModel(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(10, output_dim) # A dummy layer to have parameters
        self.forward_called_count = 0
        self.last_input = None
        self._training_mode = True # To check model.train()/eval()
        self.current_device = torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_called_count += 1
        self.last_input = x
        batch_size = x.shape[0]
        # Simulate logits output, e.g., alternating 0.7 and -0.7 for varied sigmoid outputs
        output = torch.tensor([[0.7] if i % 2 == 0 else [-0.7] for i in range(batch_size)], 
                                dtype=torch.float32, device=x.device)
        return output.view(batch_size, self.output_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self._training_mode = mode
        return self

    def eval(self):
        super().eval()
        self._training_mode = False
        return self
    
    def to(self, device):
        self.current_device = device
        return super().to(device)

class MockOptimizer:
    def __init__(self, params):
        self.params = list(params)
        self.zero_grad_called_count = 0
        self.step_called_count = 0

    def zero_grad(self):
        self.zero_grad_called_count += 1

    def step(self):
        self.step_called_count += 1

# --- Pytest Fixtures ---
@pytest.fixture
def mock_model() -> MockModel:
    return MockModel(output_dim=1)

@pytest.fixture
def mock_optimizer(mock_model: MockModel) -> MockOptimizer:
    return MockOptimizer(mock_model.parameters())

@pytest.fixture
def criterion() -> nn.Module:
    return nn.BCEWithLogitsLoss()

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_dataloader() -> DataLoader:
    # seq_len = 10, 2 batches of size 2
    texts = torch.randint(0, 100, (4, 10), dtype=torch.long)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long) # For BCEWithLogitsLoss, trainer converts to float().unsqueeze(1)
    dataset = TensorDataset(texts, labels)
    return DataLoader(dataset, batch_size=2)

@pytest.fixture
def dummy_prediction_dataloader() -> DataLoader:
    texts = torch.randint(0, 100, (3, 10), dtype=torch.long) # 3 samples
    dataset = TensorDataset(texts) # Note the comma is not needed for single tensor
    return DataLoader(dataset, batch_size=2) # Batches of 2 and 1

@pytest.fixture
def trainer(mock_model: MockModel, criterion: nn.Module, mock_optimizer: MockOptimizer, device: torch.device) -> ModelTrainer:
    return ModelTrainer(mock_model, criterion, mock_optimizer, device)

# --- Test Cases for ModelTrainer ---

def test_trainer_initialization(trainer: ModelTrainer, mock_model: MockModel, criterion: nn.Module, device: torch.device):
    assert trainer.model == mock_model
    assert trainer.criterion == criterion
    assert trainer.optimizer is not None
    assert trainer.device == device
    assert mock_model.current_device == device # Check if model was moved to device
    # Criterion is also moved to device by trainer
    # This is harder to check directly without inspecting internal state of BCEWithLogitsLoss, 
    # but we trust PyTorch's .to() method call.

def test_binary_accuracy(trainer: ModelTrainer):
    # Test with logits that would become [1, 0, 1, 0] after sigmoid and round
    # sigmoid(2) approx 0.88 -> 1; sigmoid(-2) approx 0.12 -> 0
    preds = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]], device=trainer.device) 
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]], device=trainer.device)
    accuracy = trainer._binary_accuracy(preds, labels)
    assert accuracy == 1.0

    preds_half_wrong = torch.tensor([[2.0], [2.0], [-2.0], [-2.0]], device=trainer.device)
    # True labels:                      1      0       1        0
    # Rounded preds:                    1      1       0        0 
    # Correct:                          Yes    No      No       Yes (2 out of 4 correct)
    accuracy_half = trainer._binary_accuracy(preds_half_wrong, labels)
    assert accuracy_half == 0.5

def test_train_epoch(trainer: ModelTrainer, mock_model: MockModel, mock_optimizer: MockOptimizer, dummy_dataloader: DataLoader):
    mock_model.forward_called_count = 0 # Reset counter
    mock_optimizer.zero_grad_called_count = 0
    mock_optimizer.step_called_count = 0
    
    initial_model_training_state = mock_model._training_mode

    epoch_loss, epoch_acc = trainer.train_epoch(dummy_dataloader)

    assert mock_model._training_mode is True # Check model.train() was effectively called
    assert mock_model.forward_called_count == len(dummy_dataloader) # Called for each batch
    assert mock_optimizer.zero_grad_called_count == len(dummy_dataloader)
    assert mock_optimizer.step_called_count == len(dummy_dataloader)
    
    assert isinstance(epoch_loss, float)
    assert isinstance(epoch_acc, float)
    assert 0.0 <= epoch_acc <= 1.0
    # Loss can be anything, but should be a float

def test_evaluate_epoch(trainer: ModelTrainer, mock_model: MockModel, dummy_dataloader: DataLoader):
    mock_model.forward_called_count = 0 # Reset counter
    initial_model_training_state = mock_model._training_mode
    
    # Ensure model starts in a known state (e.g., train) before evaluate forces eval mode
    mock_model.train()
    assert mock_model._training_mode is True

    epoch_loss, epoch_acc = trainer.evaluate(dummy_dataloader)

    assert mock_model._training_mode is False # Check model.eval() was effectively called
    assert mock_model.forward_called_count == len(dummy_dataloader) # Called for each batch
    
    assert isinstance(epoch_loss, float)
    assert isinstance(epoch_acc, float)
    assert 0.0 <= epoch_acc <= 1.0

def test_predict_raw_scores(trainer: ModelTrainer, mock_model: MockModel, dummy_prediction_dataloader: DataLoader):
    mock_model.forward_called_count = 0
    mock_model.train() # Set to train to ensure predict changes to eval

    scores = trainer.predict_raw_scores(dummy_prediction_dataloader)
    
    assert mock_model._training_mode is False # Ensure model.eval() was called
    assert mock_model.forward_called_count == len(dummy_prediction_dataloader)
    assert isinstance(scores, list)
    assert len(scores) == sum(len(batch_texts[0]) for batch_texts in dummy_prediction_dataloader) # Total number of samples
    for score in scores:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0 # Sigmoid output

    # Test with single item batch that might not return a list from tolist()
    single_item_tensor = torch.randint(0,100, (1,5), dtype=torch.long)
    single_item_dataset = TensorDataset(single_item_tensor)
    single_item_loader = DataLoader(single_item_dataset, batch_size=1)
    single_scores = trainer.predict_raw_scores(single_item_loader)
    assert len(single_scores) == 1
    assert isinstance(single_scores[0], float)

# --- Test Cases for convert_score_to_categorical ---
@pytest.mark.parametrize("score, expected_category", [
    (0.8, 1),   # Positive
    (0.67, 1),  # Positive (just above threshold)
    (0.66, 0),  # Neutral (at positive_threshold, inclusive for neutral by PRD logic)
    (0.5, 0),   # Neutral
    (0.33, 0),  # Neutral (at negative_threshold, inclusive for neutral)
    (0.32, -1), # Negative (just below threshold)
    (0.1, -1),  # Negative
    (0.0, -1),  # Negative (score == 0.0)
    (1.0, 1),   # Positive (score == 1.0)
])
def test_convert_score_to_categorical_valid_range(score: float, expected_category: int):
    assert convert_score_to_categorical(score) == expected_category

@pytest.mark.parametrize("score, expected_category_after_clamping", [
    (-0.5, -1), # Clamped to 0.0, then categorized as Negative
    (1.5, 1),   # Clamped to 1.0, then categorized as Positive
])
def test_convert_score_to_categorical_outside_range(score: float, expected_category_after_clamping: int):
    with pytest.warns(UserWarning, match="Score .* is outside the expected 0-1 range. Clamping"):
        assert convert_score_to_categorical(score) == expected_category_after_clamping

def test_convert_score_to_categorical_custom_thresholds():
    assert convert_score_to_categorical(0.75, positive_threshold=0.8, negative_threshold=0.4) == 0 # Neutral
    assert convert_score_to_categorical(0.85, positive_threshold=0.8, negative_threshold=0.4) == 1 # Positive
    assert convert_score_to_categorical(0.35, positive_threshold=0.8, negative_threshold=0.4) == -1# Negative 