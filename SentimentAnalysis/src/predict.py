"""
模型預測模塊。
"""
import argparse
import json
import torch

from typing import Dict, List, Union

from src.data import tokenize_chinese
from src.models import RNNSentimentClassifier
from src.train import predict


def load_model(
    model_path: str,
    vocab_path: str,
    model_config: Dict,
    device: torch.device
) -> RNNSentimentClassifier:
    """
    加載模型。
    
    Args:
        model_path: 模型路徑
        vocab_path: 詞彙表路徑
        model_config: 模型配置
        device: 設備
        
    Returns:
        加載好的模型
    """
    # 加載詞彙表
    vocab = torch.load(vocab_path)
    
    # 創建模型
    model = RNNSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["output_dim"],
        n_layers=model_config["n_layers"],
        bidirectional=model_config["bidirectional"],
        dropout=model_config["dropout"],
        rnn_type=model_config["rnn_type"],
        pad_idx=vocab["<pad>"],
    ).to(device)
    
    # 加載模型參數
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model, vocab


def predict_sentiment(
    model: RNNSentimentClassifier,
    vocab,
    texts: List[str],
    device: torch.device,
    label_map: Dict[int, str] = None
) -> List[Union[int, str]]:
    """
    預測文本情緒。
    
    Args:
        model: 模型
        vocab: 詞彙表
        texts: 文本列表
        device: 設備
        label_map: 標籤映射，將數字標籤映射到情緒標籤
        
    Returns:
        預測結果列表
    """
    predictions = predict(model, texts, vocab, tokenize_chinese, device)
    
    if label_map:
        return [label_map[p] for p in predictions]
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment of texts")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--vocab_path", required=True, help="Path to the vocabulary")
    parser.add_argument("--config_path", required=True, help="Path to the model configuration")
    parser.add_argument("--texts", nargs="+", help="Texts to predict")
    parser.add_argument("--input_file", help="Path to a file containing texts, one per line")
    parser.add_argument("--output_file", help="Path to save predictions")
    parser.add_argument("--label_map_path", help="Path to the label mapping JSON file")
    args = parser.parse_args()
    
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加載模型配置
    with open(args.config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    
    # 加載模型
    model, vocab = load_model(args.model_path, args.vocab_path, model_config, device)
    
    # 加載標籤映射（如果有）
    label_map = None
    if args.label_map_path:
        with open(args.label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    
    # 獲取要預測的文本
    texts = []
    if args.texts:
        texts = args.texts
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines()]
    else:
        raise ValueError("請提供要預測的文本，使用 --texts 或 --input_file")
    
    # 預測情緒
    predictions = predict_sentiment(model, vocab, texts, device, label_map)
    
    # 輸出預測結果
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for text, pred in zip(texts, predictions):
                f.write(f"{text}\t{pred}\n")
    else:
        for text, pred in zip(texts, predictions):
            print(f"文本: {text}")
            print(f"情緒: {pred}")
            print("---") 