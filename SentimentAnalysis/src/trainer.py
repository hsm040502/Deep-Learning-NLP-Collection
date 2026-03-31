import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import accuracy_score

# Configure logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training, evaluation, and prediction for a PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> None:
        """Initializes the ModelTrainer.

        Args:
            model: The PyTorch model to train.
            criterion: The loss function.
            optimizer: The optimizer.
            device: The device to run training/evaluation on (e.g., 'cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device

    def _binary_accuracy(self, preds: torch.Tensor, y: torch.Tensor) -> float:
        """Calculates accuracy for binary classification.

        Args:
            preds: Predictions from the model (logits or probabilities after sigmoid).
            y: True labels.

        Returns:
            Accuracy score.
        """
        # Apply sigmoid and round to get binary predictions (0 or 1)
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # Convert boolean to float
        acc = correct.sum() / len(correct)
        return acc.item()

    def train_epoch(
        self, 
        iterator: DataLoader
    ) -> Tuple[float, float]:
        """Trains the model for one epoch.

        Args:
            iterator: DataLoader for the training data.

        Returns:
            A tuple containing the epoch loss and epoch accuracy.
        """
        epoch_loss = 0.0
        epoch_acc = 0.0
        self.model.train() # Set model to training mode

        for batch_idx, (texts, labels) in enumerate(iterator):
            texts = texts.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1) # Ensure labels are float and correct shape for BCEWithLogitsLoss

            self.optimizer.zero_grad() # Clear gradients
            
            predictions = self.model(texts) # Forward pass
            
            loss = self.criterion(predictions, labels) # Calculate loss
            acc = self._binary_accuracy(predictions, labels) # Calculate accuracy
            
            loss.backward() # Backward pass
            self.optimizer.step() # Update weights
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
            if batch_idx % 10 == 0: # Log every 10 batches
                logger.debug(f"Batch {batch_idx}/{len(iterator)}: Loss={loss.item():.4f}, Acc={acc:.4f}")

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(
        self,
        iterator: DataLoader
    ) -> Tuple[float, float]:
        """Evaluates the model on a given dataset.

        Args:
            iterator: DataLoader for the evaluation data.

        Returns:
            A tuple containing the evaluation loss and evaluation accuracy.
        """
        epoch_loss = 0.0
        epoch_acc = 0.0
        self.model.eval() # Set model to evaluation mode

        with torch.no_grad(): # Disable gradient calculations
            for texts, labels in iterator:
                texts = texts.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                predictions = self.model(texts)
                
                loss = self.criterion(predictions, labels)
                acc = self._binary_accuracy(predictions, labels)

                epoch_loss += loss.item()
                epoch_acc += acc
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def predict_raw_scores(self, dataloader: DataLoader) -> List[float]:
        """Generates raw sentiment scores (0.0 to 1.0) for input data.

        Args:
            dataloader: DataLoader for the data to predict on. 
                        Assumes DataLoader yields only text sequences (no labels).

        Returns:
            A list of sentiment scores (probabilities).
        """
        self.model.eval()
        all_scores: List[float] = []
        with torch.no_grad():
            for (texts,) in dataloader: # Expecting only texts from dataloader for prediction
                texts = texts.to(self.device)
                logits = self.model(texts)
                scores = torch.sigmoid(logits).squeeze().cpu().tolist()
                if isinstance(scores, float): # Handle single item batch
                    all_scores.append(scores)
                else:
                    all_scores.extend(scores)
        return all_scores

def convert_score_to_categorical(
    score: float, 
    positive_threshold: float = 0.66, 
    negative_threshold: float = 0.33
) -> int:
    """Converts a sentiment score (0.0-1.0) to a categorical label.

    As per PRD section 3.4:
    - Positive (1): score > 0.66
    - Neutral (0): 0.33 <= score <= 0.66
    - Negative (-1): score < 0.33

    Args:
        score: The sentiment score, between 0.0 and 1.0.
        positive_threshold: Threshold above which sentiment is considered positive.
        negative_threshold: Threshold below which sentiment is considered negative.
                            (Neutral is between negative_threshold and positive_threshold inclusive)

    Returns:
        Categorical sentiment label: 1 for positive, 0 for neutral, -1 for negative.
    """
    if not (0.0 <= score <= 1.0):
        logger.warning(f"Score {score} is outside the expected 0-1 range. Clamping for categorization.")
        score = max(0.0, min(1.0, score))

    if score > positive_threshold:
        return 1  # Positive
    elif score < negative_threshold:
        return -1 # Negative
    else:
        return 0  # Neutral


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing ModelTrainer...")

    # Mock dependencies for testing
    from src.model import SentimentLSTM # Assuming model.py is in src
    from src.preprocessing import TextPreprocessor # Assuming preprocessing.py is in src

    # 1. Setup parameters and data
    VOCAB_SIZE_TEST = 100
    EMBEDDING_DIM_TEST = 10
    HIDDEN_DIM_TEST = 20
    OUTPUT_DIM_TEST = 1
    N_LAYERS_TEST = 1
    BIDIRECTIONAL_TEST = False
    DROPOUT_RATE_TEST = 0.2
    PAD_IDX_TEST = 0
    MAX_LEN_TEST = 15
    BATCH_SIZE_TEST = 2
    N_EPOCHS_TEST = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dummy data and preprocessor
    sample_texts_train = [
        "this is a great test", "very good experience", 
        "awful product waste of money", "terrible and bad"
    ]
    sample_labels_train_str = ["positive", "positive", "negative", "negative"]
    
    sample_texts_eval = ["another good one", "not bad at all", "quite disappointing"]
    sample_labels_eval_str = ["positive", "positive", "negative"]

    preprocessor = TextPreprocessor(max_vocab_size=VOCAB_SIZE_TEST, min_freq=1, max_len=MAX_LEN_TEST)
    preprocessor.fit_on_texts(sample_texts_train + sample_texts_eval)
    
    # Update vocab size based on actual preprocessor vocab
    VOCAB_SIZE_TEST = preprocessor.vocab_size 
    PAD_IDX_TEST = preprocessor.pad_idx

    train_sequences = torch.tensor(preprocessor.transform_texts(sample_texts_train), dtype=torch.long)
    train_labels = torch.tensor(preprocessor.transform_labels(sample_labels_train_str), dtype=torch.long)
    eval_sequences = torch.tensor(preprocessor.transform_texts(sample_texts_eval), dtype=torch.long)
    eval_labels = torch.tensor(preprocessor.transform_labels(sample_labels_eval_str), dtype=torch.long)

    train_dataset = TensorDataset(train_sequences, train_labels)
    eval_dataset = TensorDataset(eval_sequences, eval_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE_TEST)
    
    # For prediction test (no labels)
    predict_texts_raw = ["this is a new sentence for prediction", "a truly bad one"]
    predict_sequences = torch.tensor(preprocessor.transform_texts(predict_texts_raw), dtype=torch.long)
    predict_dataset = TensorDataset(predict_sequences) # Note: TensorDataset expects tuple, so (predict_sequences,)
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE_TEST)


    # 2. Initialize model, criterion, optimizer
    test_model = SentimentLSTM(
        vocab_size=VOCAB_SIZE_TEST,
        embedding_dim=EMBEDDING_DIM_TEST,
        hidden_dim=HIDDEN_DIM_TEST,
        output_dim=OUTPUT_DIM_TEST,
        n_layers=N_LAYERS_TEST,
        bidirectional=BIDIRECTIONAL_TEST,
        dropout_rate=DROPOUT_RATE_TEST,
        pad_idx=PAD_IDX_TEST
    )
    test_criterion = nn.BCEWithLogitsLoss()
    test_optimizer = optim.Adam(test_model.parameters())

    # 3. Initialize trainer
    trainer = ModelTrainer(test_model, test_criterion, test_optimizer, device)
    logger.info("ModelTrainer initialized.")

    # 4. Test training and evaluation
    logger.info("Starting dummy training...")
    for epoch in range(N_EPOCHS_TEST):
        train_loss, train_acc = trainer.train_epoch(train_dataloader)
        eval_loss, eval_acc = trainer.evaluate(eval_dataloader)
        logger.info(f"Epoch {epoch+1}/{N_EPOCHS_TEST}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        logger.info(f"  Eval Loss:  {eval_loss:.4f}, Eval Acc:  {eval_acc*100:.2f}%")

    # 5. Test prediction
    logger.info("\nTesting prediction...")
    raw_scores = trainer.predict_raw_scores(predict_dataloader)
    logger.info(f"Raw predicted scores: {raw_scores}")

    # 6. Test score to category conversion
    logger.info("\nTesting score to category conversion:")
    test_scores_for_conversion = [0.1, 0.33, 0.5, 0.66, 0.8, 0.0, 1.0, -0.1, 1.1]
    for s in test_scores_for_conversion:
        cat = convert_score_to_categorical(s)
        logger.info(f"Score: {s:.2f} -> Category: {cat} ({'Negative' if cat == -1 else 'Neutral' if cat == 0 else 'Positive'})")

    logger.info("\nModelTrainer test finished.") 