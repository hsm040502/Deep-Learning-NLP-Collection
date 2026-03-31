import logging
import nltk
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Special token names
PAD_TOKEN = "<PAD>"  # Padding token
UNK_TOKEN = "<UNK>"  # Unknown token

def download_nltk_resource_if_not_exists(resource_id: str, download_name: str) -> None:
    """Downloads an NLTK resource if it's not already present.

    Args:
        resource_id: The NLTK resource identifier (e.g., 'tokenizers/punkt').
        download_name: The name to use for nltk.download() (e.g., 'punkt').
    """
    try:
        nltk.data.find(resource_id)
        logger.debug(f"NLTK resource '{download_name}' already downloaded.")
    except nltk.downloader.DownloadError:
        logger.info(f"NLTK resource '{download_name}' not found. Downloading...")
        try:
            nltk.download(download_name, quiet=True)
            logger.info(f"NLTK resource '{download_name}' downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource '{download_name}'. Error: {e}")
            logger.error(
                f"Please try manually downloading by running: import nltk; nltk.download('{download_name}')"
            )
            # Depending on strictness, might raise an error here
            # For now, we'll log and proceed, tokenization might fail later.

class TextPreprocessor:
    """Handles text preprocessing: tokenization, vocabulary building, numericalization, and padding."""

    def __init__(
        self,
        max_vocab_size: Optional[int] = 20000,
        min_freq: int = 1,
        max_len: Optional[int] = None,
    ) -> None:
        """Initializes the TextPreprocessor.

        Args:
            max_vocab_size: The maximum number of words to keep in the vocabulary,
                            based on frequency. If None, all words (respecting min_freq)
                            are kept.
            min_freq: The minimum frequency a word must have to be included in the
                      vocabulary.
            max_len: The fixed length to pad/truncate sequences to. If None, it will be
                     determined by the longest sequence encountered during fit_on_texts,
                     or can be set later before padding.
        """
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.max_len: Optional[int] = max_len
        self.pad_token: str = PAD_TOKEN
        self.unk_token: str = UNK_TOKEN
        self.pad_idx: int = 0  # Will be set after vocab is built
        self.unk_idx: int = 1  # Will be set after vocab is built
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # Ensure NLTK's sentence tokenizer (punkt) is available
        download_nltk_resource_if_not_exists("tokenizers/punkt", "punkt")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes a single text string using NLTK.

        Args:
            text: The input string.

        Returns:
            A list of tokens.
        """
        try:
            return nltk.word_tokenize(text.lower()) # Convert to lowercase
        except Exception as e:
            logger.error(f"Error during NLTK word_tokenize for text '{text[:50]}...': {e}")
            return text.lower().split() # Fallback to simple split

    def fit_on_texts(self, texts: List[str]) -> None:
        """Builds the vocabulary from a list of texts.

        Args:
            texts: A list of raw text strings.
        """
        logger.info("Fitting vocabulary on texts...")
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        word_counts = Counter()
        for tokens in tokenized_texts:
            word_counts.update(tokens)

        # Sort words by frequency, then alphabetically for ties
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

        self.word_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_word = {0: self.pad_token, 1: self.unk_token}
        self.pad_idx = self.word_to_idx[self.pad_token]
        self.unk_idx = self.word_to_idx[self.unk_token]

        current_idx = 2 # Start after PAD and UNK
        for word, count in sorted_words:
            if self.max_vocab_size is not None and current_idx >= (self.max_vocab_size + 2): # +2 for PAD/UNK
                logger.info(f"Reached max_vocab_size ({self.max_vocab_size}). Ignoring remaining words.")
                break
            if count < self.min_freq:
                # Since words are sorted by frequency, we can break early
                logger.info(f"Ignoring words with frequency < {self.min_freq}. Word: '{word}' (count: {count})")
                break 
            if word not in self.word_to_idx:
                self.word_to_idx[word] = current_idx
                self.idx_to_word[current_idx] = word
                current_idx += 1
        
        self.vocab_size = len(self.word_to_idx)
        logger.info(f"Vocabulary built. Size: {self.vocab_size} words (including PAD & UNK).")

        if self.max_len is None:
            if not tokenized_texts:
                self.max_len = 0 # Or a default value, e.g. 50
            else:
                self.max_len = max(len(tokens) for tokens in tokenized_texts)
            logger.info(f"max_len was not set, determined from data: {self.max_len}")

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Converts a list of texts into sequences of numerical indices.

        Args:
            texts: A list of raw text strings.

        Returns:
            A list of lists, where each inner list is a sequence of word indices.
        """
        if not self.word_to_idx:
            raise RuntimeError("Vocabulary not built. Call fit_on_texts() first.")
        
        numerical_sequences: List[List[int]] = []
        for text in texts:
            tokens = self._tokenize(text)
            sequence = [self.word_to_idx.get(token, self.unk_idx) for token in tokens]
            numerical_sequences.append(sequence)
        return numerical_sequences

    def pad_sequences(self, sequences: List[List[int]], max_len: Optional[int] = None) -> List[List[int]]:
        """Pads or truncates sequences to a fixed length.

        Args:
            sequences: A list of numerical sequences (lists of integers).
            max_len: The target length for all sequences. If None, uses self.max_len.

        Returns:
            A list of padded/truncated sequences.
        """
        target_len = max_len if max_len is not None else self.max_len
        if target_len is None:
            raise ValueError("max_len must be specified either at init or in this method, or fit_on_texts must be called.")

        padded_sequences: List[List[int]] = []
        for seq in sequences:
            if len(seq) > target_len:
                padded_seq = seq[:target_len]  # Truncate
            else:
                padded_seq = seq + [self.pad_idx] * (target_len - len(seq))  # Pad
            padded_sequences.append(padded_seq)
        return padded_sequences

    def transform_texts(self, texts: List[str], max_len: Optional[int] = None) -> List[List[int]]:
        """Combines texts_to_sequences and pad_sequences.

        Args:
            texts: A list of raw text strings.
            max_len: Optional target length for padding. Uses self.max_len if None.

        Returns:
            A list of processed (numericalized and padded) sequences.
        """
        sequences = self.texts_to_sequences(texts)
        padded_sequences = self.pad_sequences(sequences, max_len)
        return padded_sequences

    def transform_labels(self, labels: List[str]) -> List[int]:
        """Converts string labels ('positive', 'negative') to numerical labels (1, 0).

        As per PRD section 3.4, true labels are 'positive' (1) and 'negative' (-1).
        However, for training binary classification models (e.g., with sigmoid output
        and BCEWithLogitsLoss), labels are typically 0 and 1.
        This function converts to 0/1 for training convenience.
        The final CLI output can handle mapping back to 1/-1/0 for display.

        Args:
            labels: A list of string labels (e.g., ['positive', 'negative']).

        Returns:
            A list of numerical labels (e.g., [1, 0]).
        """
        numerical_labels: List[int] = []
        for label in labels:
            if label.lower() == "positive":
                numerical_labels.append(1)
            elif label.lower() == "negative":
                numerical_labels.append(0)
            else:
                logger.warning(f"Unknown label '{label}' encountered. Assigning 0. Please check data.")
                numerical_labels.append(0) # Or handle as an error
        return numerical_labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing TextPreprocessor...")

    # Sample data (in a real scenario, this would come from data_loader)
    sample_texts = [
        "This is a great product, I love it!",
        "Terrible experience, very bad.",
        "Just okay, not great but not bad either.",
        "Another great comment for testing purposes.",
        "I hate this product, it is the worst!",
    ]
    sample_labels = ["positive", "negative", "positive", "positive", "negative"] # Changed middle to positive for balance

    # 1. Initialize preprocessor
    # Let max_len be determined automatically for this test
    preprocessor = TextPreprocessor(max_vocab_size=100, min_freq=1, max_len=None)
    logger.info(f"Preprocessor initialized. Initial max_len: {preprocessor.max_len}")

    # 2. Fit on texts to build vocabulary
    preprocessor.fit_on_texts(sample_texts)
    logger.info(f"Vocabulary: {preprocessor.word_to_idx}")
    logger.info(f"Determined max_len: {preprocessor.max_len}")

    # 3. Transform texts
    processed_texts = preprocessor.transform_texts(sample_texts)
    logger.info("Processed Texts (first 2):")
    for i in range(min(2, len(processed_texts))):
        logger.info(f"  Original: {sample_texts[i]}")
        logger.info(f"  Tokens: {preprocessor._tokenize(sample_texts[i])}") # Show tokenization
        logger.info(f"  Sequence: {preprocessor.texts_to_sequences([sample_texts[i]])[0]}") # Show unpadded sequence
        logger.info(f"  Padded:   {processed_texts[i]}")

    # 4. Transform labels
    processed_labels = preprocessor.transform_labels(sample_labels)
    logger.info("Processed Labels (first 5):")
    for i in range(min(5, len(processed_labels))):
        logger.info(f"  Original: {sample_labels[i]}, Processed: {processed_labels[i]}")

    logger.info("TextPreprocessor test finished.")

    logger.info("\nDemonstrating with max_len explicitly set and unknown words:")
    preprocessor_fixed_len = TextPreprocessor(max_vocab_size=10, min_freq=1, max_len=10)
    texts_for_fixed_len = [
        "short example a a a", 
        "this is a much much much longer example with new words here there everywhere"
    ]
    preprocessor_fixed_len.fit_on_texts(texts_for_fixed_len) # Fit on its own small corpus
    logger.info(f"Fixed_len Vocab: {preprocessor_fixed_len.word_to_idx}")
    transformed_fixed_len = preprocessor_fixed_len.transform_texts(
        texts_for_fixed_len + ["new unknown text another one"] # Add text with OOV words
    )
    logger.info("Transformed texts with fixed_len=10 and OOV:")
    for i, original_text in enumerate(texts_for_fixed_len + ["new unknown text another one"]):
        logger.info(f"  Original: {original_text}")
        logger.info(f"  Padded:   {transformed_fixed_len[i]}") 