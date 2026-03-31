import logging
import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from src.data_loader import load_reviews_from_xml, ReviewParseError
from src.preprocessing import TextPreprocessor
from src.model import SentimentLSTM
from src.trainer import ModelTrainer, convert_score_to_categorical

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__) # Get a logger for this module

# --- Configuration & Hyperparameters ---
# These could be moved to a config file (e.g., YAML) or command-line arguments
DATA_FILE_PATH = Path("data/Comments_Mobile_Phone.xml")

# Model Hyperparameters
VOCAB_SIZE_MAX = 20000
MIN_FREQ = 2
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 1 # For binary classification (outputting a single logit)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT_RATE = 0.5
# MAX_LEN will be determined by TextPreprocessor or can be set manually
# MAX_LEN = 200 # Example: If you want to fix max sequence length

# Training Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
N_EPOCHS = 10 # Number of training epochs

# Thresholds for sentiment categorization (as per PRD)
POSITIVE_THRESHOLD = 0.66
NEGATIVE_THRESHOLD = 0.33

def map_original_label_to_int(label: str) -> int:
    """Maps original string labels to integer representation as per PRD (1 for positive, -1 for negative)."""
    if label.lower() == "positive":
        return 1
    elif label.lower() == "negative":
        return -1
    logger.warning(f"Unknown original label '{label}' found. Defaulting to 0 (neutral) for display purposes.")
    return 0 # Should ideally not happen if data is clean

def get_sentiment_text(categorical_label: int) -> str:
    """Converts categorical label (1, 0, -1) to text."""
    if categorical_label == 1:
        return "Positive"
    elif categorical_label == -1:
        return "Negative"
    elif categorical_label == 0:
        return "Neutral"
    return "Unknown"

def main_cli() -> None:
    """Main command-line interface function for the sentiment analysis task."""
    logger.info("Starting Sentiment Analysis CLI Application...")
    start_time = time.time()

    # --- 0. Setup --- 
    parser = argparse.ArgumentParser(description="RNN-based Sentiment Analysis CLI")
    parser.add_argument("--data_file", type=str, default=str(DATA_FILE_PATH), help="Path to the XML data file.")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training and prediction.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate for the optimizer.")
    # Add more arguments for other hyperparameters if needed

    args = parser.parse_args()
    current_data_file_path = Path(args.data_file)
    current_n_epochs = args.epochs
    current_batch_size = args.batch_size
    current_lr = args.lr

    logger.info(f"Using data file: {current_data_file_path}")
    logger.info(f"Number of epochs: {current_n_epochs}")
    logger.info(f"Batch size: {current_batch_size}")
    logger.info(f"Learning rate: {current_lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. Data Loading & Preprocessing ---
    logger.info("Loading and preprocessing data...")
    try:
        raw_reviews_with_labels = load_reviews_from_xml(current_data_file_path)
        if not raw_reviews_with_labels:
            logger.error(f"No reviews loaded from {current_data_file_path}. Exiting.")
            return
    except FileNotFoundError:
        logger.error(f"Data file {current_data_file_path} not found. Please ensure it exists. Exiting.")
        return
    except ReviewParseError as e:
        logger.error(f"Error parsing XML file: {e}. Exiting.")
        return

    all_texts = [text for text, label in raw_reviews_with_labels]
    original_labels_str = [label for text, label in raw_reviews_with_labels]
    
    # Store original texts for final output
    original_texts_for_output = list(all_texts)
    
    # Convert original string labels to PRD-defined integers (1 for pos, -1 for neg)
    # These are the "true" sentiment scores/categories for comparison.
    true_sentiment_categories = [map_original_label_to_int(label) for label in original_labels_str]

    preprocessor = TextPreprocessor(max_vocab_size=VOCAB_SIZE_MAX, min_freq=MIN_FREQ) # max_len can be auto-determined
    preprocessor.fit_on_texts(all_texts)
    
    # Important: Get actual vocab_size and pad_idx from the preprocessor after fitting
    actual_vocab_size = preprocessor.vocab_size
    pad_idx = preprocessor.pad_idx
    max_len_determined = preprocessor.max_len
    logger.info(f"Vocabulary size: {actual_vocab_size}, Padding index: {pad_idx}, Max sequence length: {max_len_determined}")

    # Prepare data for training the model (using 0/1 labels for BCEWithLogitsLoss)
    training_labels_numeric = preprocessor.transform_labels(original_labels_str) # Converts to 0 (neg) / 1 (pos)
    processed_sequences = preprocessor.transform_texts(all_texts) # Texts are numericalized and padded

    # Convert to PyTorch tensors
    sequences_tensor = torch.tensor(processed_sequences, dtype=torch.long)
    labels_tensor = torch.tensor(training_labels_numeric, dtype=torch.long)

    # Create DataLoader for training (and later for prediction on the same full dataset as per PRD)
    # PRD: "analyse data/Comments_Mobile_Phone.xml中的評論... 在所有評論分析完成後，計算並印出整個模型在該資料集上的「準確度 (Accuracy)」"
    # This implies training and then evaluating/predicting on the full dataset for this specific project scope.
    # For a more general setup, one would split into train/validation/test sets.
    full_dataset = TensorDataset(sequences_tensor, labels_tensor)
    # Shuffling for training is good practice, even if we evaluate on the same data afterwards
    train_dataloader = DataLoader(full_dataset, batch_size=current_batch_size, shuffle=True)
    # For prediction/final output, we'll iterate without shuffling to match original order
    # We will need a dataloader that only yields texts for predict_raw_scores
    prediction_dataset = TensorDataset(sequences_tensor) # Only sequences for prediction
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=current_batch_size, shuffle=False)


    # --- 2. Model Preparation & Training ---
    logger.info("Initializing model, optimizer, and criterion...")
    model = SentimentLSTM(
        vocab_size=actual_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout_rate=DROPOUT_RATE,
        pad_idx=pad_idx,
    )
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification with logits output

    trainer = ModelTrainer(model, criterion, optimizer, device)

    logger.info(f"Starting model training for {current_n_epochs} epochs...")
    for epoch in range(current_n_epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = trainer.train_epoch(train_dataloader)
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{current_n_epochs} completed in {epoch_duration:.2f}s")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
        # As per PRD, the final evaluation is on the whole dataset after training is complete.
        # If a validation set were used, it would be evaluated here.

    logger.info("Model training finished.")

    # --- 3. Prediction & Evaluation on the full dataset ---
    logger.info("Predicting sentiment scores for all loaded reviews...")
    # Ensure the model is in evaluation mode for predictions
    model.eval()
    predicted_raw_scores = trainer.predict_raw_scores(prediction_dataloader) # Uses the model in trainer

    if len(predicted_raw_scores) != len(original_texts_for_output):
        logger.error(
            f"Mismatch in number of predictions ({len(predicted_raw_scores)}) "
            f"and original texts ({len(original_texts_for_output)}). Cannot proceed with output."
        )
        return

    predicted_categorical_labels = [
        convert_score_to_categorical(score, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD)
        for score in predicted_raw_scores
    ]

    # --- 4. Results Presentation (as per PRD) ---
    logger.info("\n--- Sentiment Analysis Results ---")
    num_correct_predictions = 0
    for i in range(len(original_texts_for_output)):
        original_text = original_texts_for_output[i]
        true_category = true_sentiment_categories[i]
        predicted_category = predicted_categorical_labels[i]
        raw_score = predicted_raw_scores[i]

        # PRD: "印出每則評論的「真實情感分數」與「預測情感分數」"
        # Assuming "真實情感分數" refers to the 1/-1 mapping and "預測情感分數" to the 1/0/-1 mapping.
        print(f"\nReview {i+1}: {original_text[:100]}..." if len(original_text) > 100 else f"\nReview {i+1}: {original_text}")
        print(f"  Raw Predicted Score: {raw_score:.4f}")
        print(f"  True Sentiment:     {true_category} ({get_sentiment_text(true_category)})")
        print(f"  Predicted Sentiment: {predicted_category} ({get_sentiment_text(predicted_category)})")

        if true_category == predicted_category:
            num_correct_predictions += 1
    
    overall_accuracy = (num_correct_predictions / len(original_texts_for_output)) * 100 if original_texts_for_output else 0
    logger.info("\n------------------------------------")
    logger.info(f"Overall Model Accuracy on the dataset: {overall_accuracy:.2f}% ({num_correct_predictions}/{len(original_texts_for_output)})")
    logger.info("------------------------------------")

    total_duration = time.time() - start_time
    logger.info(f"Sentiment Analysis CLI Application finished in {total_duration:.2f} seconds.")


if __name__ == "__main__":
    # This allows running the script directly, e.g., `python src/main.py`
    # The `analyze-sentiments` script in pyproject.toml will also call this function.
    main_cli() 