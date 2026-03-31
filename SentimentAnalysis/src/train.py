"""
模型訓練和評估模塊。
"""
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import RNNSentimentClassifier


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip: float = 1.0,
) -> Tuple[float, float]:
    """
    訓練模型一個周期。
    
    Args:
        model: 模型
        train_dataloader: 訓練數據加載器
        optimizer: 優化器
        criterion: 損失函數
        device: 設備
        clip: 梯度裁剪閾值
        
    Returns:
        (平均損失, 準確率)
    """
    model.train()
    
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(train_dataloader, desc="Training"):
        # 獲取批次數據
        text, text_lengths, labels = batch
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向傳播
        predictions, _ = model(text, text_lengths)
        
        # 計算損失
        loss = criterion(predictions, labels)
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # 更新參數
        optimizer.step()
        
        # 計算準確率
        predicted_classes = predictions.argmax(dim=1)
        correct_predictions = (predicted_classes == labels).float()
        accuracy = correct_predictions.sum() / len(correct_predictions)
        
        epoch_loss += loss.item()
        epoch_acc += accuracy.item()
    
    # 計算平均損失和準確率
    return epoch_loss / len(train_dataloader), epoch_acc / len(train_dataloader)


def evaluate_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    評估模型。
    
    Args:
        model: 模型
        test_dataloader: 測試數據加載器
        criterion: 損失函數
        device: 設備
        
    Returns:
        (平均損失, 準確率)
    """
    model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # 獲取批次數據
            text, text_lengths, labels = batch
            
            # 前向傳播
            predictions, _ = model(text, text_lengths)
            
            # 計算損失
            loss = criterion(predictions, labels)
            
            # 計算準確率
            predicted_classes = predictions.argmax(dim=1)
            correct_predictions = (predicted_classes == labels).float()
            accuracy = correct_predictions.sum() / len(correct_predictions)
            
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
    
    # 計算平均損失和準確率
    return epoch_loss / len(test_dataloader), epoch_acc / len(test_dataloader)


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    device: torch.device,
    model_save_path: str,
    patience: int = 5,
    clip: float = 1.0,
) -> Dict[str, List[float]]:
    """
    訓練模型多個周期。
    
    Args:
        model: 模型
        train_dataloader: 訓練數據加載器
        test_dataloader: 測試數據加載器
        optimizer: 優化器
        criterion: 損失函數
        n_epochs: 訓練周期數
        device: 設備
        model_save_path: 模型保存路徑
        patience: 早停耐心值，連續多少個周期驗證損失沒有改善則停止訓練
        clip: 梯度裁剪閾值
        
    Returns:
        訓練歷史記錄
    """
    # 創建保存模型的目錄
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # 訓練一個周期
        train_loss, train_acc = train_model(
            model, train_dataloader, optimizer, criterion, device, clip
        )
        
        # 評估模型
        valid_loss, valid_acc = evaluate_model(
            model, test_dataloader, criterion, device
        )
        
        # 保存歷史記錄
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # 如果驗證損失改善，保存模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%")
        
        # 早停
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
    
    return history


def predict(
    model: nn.Module,
    text: List[str],
    vocab,
    tokenize_fn,
    device: torch.device,
) -> List[int]:
    """
    使用模型預測文本的情緒。
    
    Args:
        model: 模型
        text: 文本列表
        vocab: 詞彙表
        tokenize_fn: 分詞函數
        device: 設備
        
    Returns:
        預測的標籤列表
    """
    model.eval()
    
    # 分詞
    tokenized = [tokenize_fn(t) for t in text]
    
    # 轉換為索引
    indexed = [[vocab[token] for token in tokens] for tokens in tokenized]
    
    # 獲取長度
    lengths = torch.tensor([len(indices) for indices in indexed], dtype=torch.long)
    
    # 填充序列
    max_length = max(lengths).item()
    padded = [indices + [vocab["<pad>"]] * (max_length - len(indices)) for indices in indexed]
    
    # 轉換為張量
    text_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    
    # 預測
    with torch.no_grad():
        predictions, _ = model(text_tensor, lengths)
        predicted_classes = predictions.argmax(dim=1)
    
    return predicted_classes.cpu().numpy().tolist()


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    from src.data import create_dataloaders, load_data, tokenize_chinese
    
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--data_path", required=True, help="Path to data file")
    parser.add_argument("--text_col", default="review", help="Column name for text")
    parser.add_argument("--label_col", default="sentiment", help="Column name for label")
    parser.add_argument("--embedding_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional RNN")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--model_save_path", default="models/sentiment_model.pt", help="Path to save model")
    parser.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"], help="RNN type")
    args = parser.parse_args()
    
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加載數據
    print("Loading data...")
    train_df, test_df = load_data(
        args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    
    # 創建數據加載器
    print("Creating dataloaders...")
    train_dataloader, test_dataloader, vocab = create_dataloaders(
        train_df,
        test_df,
        text_col=args.text_col,
        label_col=args.label_col,
        batch_size=args.batch_size,
        device=device,
    )
    
    # 創建模型
    print("Creating model...")
    model = RNNSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(train_df[args.label_col].unique()),
        n_layers=args.n_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        pad_idx=vocab["<pad>"],
    ).to(device)
    
    # 設定優化器和損失函數
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 訓練模型
    print("Training model...")
    history = train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        args.n_epochs,
        device,
        args.model_save_path,
        patience=args.patience,
        clip=args.clip,
    )
    
    print(f"Best model saved to {args.model_save_path}") 