import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import argparse

# Import functions and classes from src.main
# To avoid issues with logger setup at import time if main.py also configures root logger,
# we might need to be careful or ensure main_cli() is the primary entry point for such configs.
# For now, direct import should be fine if main.py's logging setup is within main_cli() or if __name__ == "__main__".
from src.main import (
    map_original_label_to_int,
    get_sentiment_text,
    main_cli,
    DATA_FILE_PATH, # For default args checking
    N_EPOCHS
)
from src.data_loader import ReviewParseError

# --- Tests for helper functions ---

def test_map_original_label_to_int():
    assert map_original_label_to_int("positive") == 1
    assert map_original_label_to_int("POSITIVE") == 1
    assert map_original_label_to_int("negative") == -1
    assert map_original_label_to_int("NEGATIVE") == -1
    with pytest.warns(UserWarning, match="Unknown original label 'neutral'"):
        assert map_original_label_to_int("neutral") == 0
    with pytest.warns(UserWarning, match="Unknown original label ''"):
        assert map_original_label_to_int("") == 0

def test_get_sentiment_text():
    assert get_sentiment_text(1) == "Positive"
    assert get_sentiment_text(0) == "Neutral"
    assert get_sentiment_text(-1) == "Negative"
    assert get_sentiment_text(5) == "Unknown"
    assert get_sentiment_text(-10) == "Unknown"

# --- Tests for main_cli (partial, using mocks) ---

@pytest.fixture
def mock_args_default() -> argparse.Namespace:
    """Returns a mock Namespace object simulating default argparse output."""
    return argparse.Namespace(
        data_file=str(DATA_FILE_PATH), # Use the default from main.py
        epochs=N_EPOCHS,               # Use the default from main.py
        batch_size=64,
        lr=1e-3
    )

@pytest.fixture
def mock_args_custom() -> argparse.Namespace:
    """Returns a mock Namespace object simulating custom argparse output."""
    return argparse.Namespace(
        data_file="custom/path/to/data.xml",
        epochs=5,
        batch_size=32,
        lr=0.01
    )

@patch("src.main.argparse.ArgumentParser.parse_args")
@patch("src.main.load_reviews_from_xml")
@patch("src.main.TextPreprocessor")
@patch("src.main.SentimentLSTM")
@patch("src.main.ModelTrainer")
@patch("src.main.torch.device") # Mock device to control CPU/GPU for test consistency
@patch("builtins.print") # Mock print to avoid console output during tests
@patch("src.main.logger") # Mock logger to check logging calls
def test_main_cli_data_loading_file_not_found(
    mock_logger: MagicMock, mock_print: MagicMock, mock_device: MagicMock, 
    mock_model_trainer: MagicMock, mock_sentiment_lstm: MagicMock, 
    mock_text_preprocessor: MagicMock, mock_load_reviews: MagicMock, 
    mock_parse_args: MagicMock, mock_args_default: argparse.Namespace
):
    """Test main_cli exits if data file is not found."""
    mock_parse_args.return_value = mock_args_default
    mock_load_reviews.side_effect = FileNotFoundError("File not found for test")
    mock_device.return_value = torch.device("cpu") # Ensure consistent device

    main_cli()

    mock_load_reviews.assert_called_once_with(Path(mock_args_default.data_file))
    mock_logger.error.assert_any_call(f"Data file {Path(mock_args_default.data_file)} not found. Please ensure it exists. Exiting.")
    # Check that subsequent major steps are NOT called
    mock_text_preprocessor.assert_not_called()
    mock_model_trainer.assert_not_called()

@patch("src.main.argparse.ArgumentParser.parse_args")
@patch("src.main.load_reviews_from_xml")
@patch("src.main.TextPreprocessor")
@patch("src.main.logger")
def test_main_cli_data_loading_no_reviews(
    mock_logger: MagicMock, mock_text_preprocessor: MagicMock, 
    mock_load_reviews: MagicMock, mock_parse_args: MagicMock, 
    mock_args_default: argparse.Namespace
):
    """Test main_cli exits if no reviews are loaded."""
    mock_parse_args.return_value = mock_args_default
    mock_load_reviews.return_value = [] # Simulate no reviews loaded
    with patch("src.main.torch.device", return_value=torch.device("cpu")):
        main_cli()

    mock_load_reviews.assert_called_once_with(Path(mock_args_default.data_file))
    mock_logger.error.assert_any_call(f"No reviews loaded from {Path(mock_args_default.data_file)}. Exiting.")
    mock_text_preprocessor.assert_not_called()

@patch("src.main.argparse.ArgumentParser.parse_args")
@patch("src.main.load_reviews_from_xml")
@patch("src.main.TextPreprocessor")
@patch("src.main.SentimentLSTM")
@patch("src.main.ModelTrainer")
@patch("src.main.DataLoader") # Mock DataLoader to avoid actual data iteration
@patch("src.main.TensorDataset") # Mock TensorDataset
@patch("src.main.torch.device")
@patch("builtins.print")
@patch("src.main.logger")
def test_main_cli_successful_run_flow_and_args_usage(
    mock_logger: MagicMock, mock_print: MagicMock, mock_device: MagicMock,
    mock_tensor_dataset: MagicMock, mock_data_loader: MagicMock, 
    mock_model_trainer_class: MagicMock, mock_sentiment_lstm_class: MagicMock, 
    mock_text_preprocessor_class: MagicMock, mock_load_reviews: MagicMock, 
    mock_parse_args: MagicMock, mock_args_custom: argparse.Namespace # Use custom args here
):
    """Test a simplified successful flow, checking if critical components are called
    and custom arguments are used.
    """
    mock_parse_args.return_value = mock_args_custom
    mock_device.return_value = torch.device("cpu")

    # --- Mock return values for components ---
    # 1. Data Loader
    sample_raw_reviews = [("good review", "positive"), ("bad review", "negative")]
    mock_load_reviews.return_value = sample_raw_reviews

    # 2. TextPreprocessor instance and its methods
    mock_preprocessor_instance = MagicMock(spec=mock_text_preprocessor_class)
    # Make vocab_size and pad_idx properties that can be set and read
    type(mock_preprocessor_instance).vocab_size = PropertyMock(return_value=100)
    type(mock_preprocessor_instance).pad_idx = PropertyMock(return_value=0)
    type(mock_preprocessor_instance).max_len = PropertyMock(return_value=20)
    mock_preprocessor_instance.transform_labels.return_value = [1, 0] 
    mock_preprocessor_instance.transform_texts.return_value = [[10, 11], [12, 13]]
    mock_text_preprocessor_class.return_value = mock_preprocessor_instance

    # 3. TensorDataset and DataLoader
    mock_tensor_dataset.return_value = MagicMock() # Dummy dataset object
    mock_data_loader.return_value = MagicMock()    # Dummy dataloader object
    # Simulate iteration for the dataloaders used in training/prediction
    # For train_dataloader (yields two items: texts, labels)
    # For prediction_dataloader (yields one item: texts)
    # This is simplified; in reality, DataLoader yields batches.
    # For this high-level test, we just ensure they are created.
    def dataloader_side_effect(*args, **kwargs):
        if len(args[0]) == 2: # Assuming this is how we distinguish dataset for training (texts, labels)
            return iter([ (torch.tensor([[1,2]]), torch.tensor([1])) ]) # Simplified batch for training
        else: # Dataset for prediction (texts only)
            return iter([ (torch.tensor([[1,2]]),) ]) # Simplified batch for prediction
    mock_data_loader.side_effect = dataloader_side_effect

    # 4. Model and Trainer
    mock_lstm_instance = MagicMock(spec=mock_sentiment_lstm_class)
    mock_sentiment_lstm_class.return_value = mock_lstm_instance
    
    mock_trainer_instance = MagicMock(spec=mock_model_trainer_class)
    mock_trainer_instance.train_epoch.return_value = (0.1, 0.9) # loss, acc
    mock_trainer_instance.predict_raw_scores.return_value = [0.95, 0.05] # Two scores for two reviews
    mock_model_trainer_class.return_value = mock_trainer_instance

    # --- Execute main_cli ---
    main_cli()

    # --- Assertions ---
    # Argument usage
    mock_parse_args.assert_called_once()
    mock_logger.info.assert_any_call(f"Using data file: {Path(mock_args_custom.data_file)}")
    mock_logger.info.assert_any_call(f"Number of epochs: {mock_args_custom.epochs}")
    mock_logger.info.assert_any_call(f"Batch size: {mock_args_custom.batch_size}")
    mock_logger.info.assert_any_call(f"Learning rate: {mock_args_custom.lr}")

    # Component calls
    mock_load_reviews.assert_called_once_with(Path(mock_args_custom.data_file))
    mock_text_preprocessor_class.assert_called_once()
    mock_preprocessor_instance.fit_on_texts.assert_called_once_with(["good review", "bad review"])
    mock_preprocessor_instance.transform_labels.assert_called_once_with(["positive", "negative"])
    mock_preprocessor_instance.transform_texts.assert_called_once_with(["good review", "bad review"])
    
    # Check DataLoader calls (one for training, one for prediction)
    # We expect two TensorDataset calls, then two DataLoader calls.
    assert mock_tensor_dataset.call_count == 2
    assert mock_data_loader.call_count == 2

    mock_sentiment_lstm_class.assert_called_once_with(
        vocab_size=100, # from mock_preprocessor_instance.vocab_size
        embedding_dim=128, # Default from main.py
        hidden_dim=256,    # Default from main.py
        output_dim=1,      # Default from main.py
        n_layers=2,        # Default from main.py
        bidirectional=True,# Default from main.py
        dropout_rate=0.5,  # Default from main.py
        pad_idx=0          # from mock_preprocessor_instance.pad_idx
    )
    mock_model_trainer_class.assert_called_once()
    # Check if trainer.train_epoch was called for the number of custom epochs
    assert mock_trainer_instance.train_epoch.call_count == mock_args_custom.epochs
    mock_trainer_instance.predict_raw_scores.assert_called_once()

    # Check print calls for results (simplified check)
    mock_print.assert_any_call("\nReview 1: good review")
    mock_print.assert_any_call("  True Sentiment:     1 (Positive)")
    mock_print.assert_any_call("  Predicted Sentiment: 1 (Positive)") # Based on score 0.95
    mock_print.assert_any_call("\nReview 2: bad review")
    mock_print.assert_any_call("  True Sentiment:     -1 (Negative)")
    mock_print.assert_any_call("  Predicted Sentiment: -1 (Negative)") # Based on score 0.05

    mock_logger.info.assert_any_call(pytest.lazy_fixture("Overall Model Accuracy on the dataset: 100.00% (2/2)"))

