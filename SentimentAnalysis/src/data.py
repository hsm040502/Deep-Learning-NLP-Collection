"""
數據處理和加載模塊。
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import jieba
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator


class ReviewDataset(Dataset):
    """
    評論數據集類。
    
    Attributes:
        texts: 文本列表
        labels: 標籤列表
    """
    
    def __init__(self, texts: List[List[str]], labels: List[int]) -> None:
        """
        初始化數據集。
        
        Args:
            texts: 經過分詞的文本列表
            labels: 標籤列表
        """
        self.texts = texts
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[List[str], int]:
        return self.texts[idx], self.labels[idx]


def tokenize_chinese(text: str) -> List[str]:
    """
    對中文文本進行分詞。
    
    Args:
        text: 待分詞的文本
        
    Returns:
        分詞後的標記列表
    """
    # 使用jieba進行中文分詞
    return list(jieba.cut(text))


def load_data(
    data_path: str,
    text_col: str = "review",
    label_col: str = "sentiment",
    split_ratio: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加載並拆分數據。
    
    Args:
        data_path: 數據文件路徑
        text_col: 文本列欄位名
        label_col: 標籤列欄位名
        split_ratio: 訓練集比例
        random_state: 隨機種子
        
    Returns:
        訓練集和測試集的DataFrame元組
    """
    # 檢查文件類型並加載
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {os.path.splitext(data_path)[1]}")
    
    # 確保必要的欄位存在
    if text_col not in df.columns:
        raise ValueError(f"數據中沒有找到文本列: {text_col}")
    if label_col not in df.columns:
        raise ValueError(f"數據中沒有找到標籤列: {label_col}")
    
    # 拆分訓練集和測試集
    train_df = df.sample(frac=split_ratio, random_state=random_state)
    test_df = df.drop(train_df.index)
    
    return train_df, test_df


def build_vocabulary(
    texts: List[List[str]], min_freq: int = 2, specials: List[str] = ["<unk>", "<pad>"]
) -> Vocab:
    """
    從文本構建詞彙表。
    
    Args:
        texts: 分詞後的文本列表
        min_freq: 最小詞頻
        specials: 特殊標記列表
        
    Returns:
        構建好的詞彙表
    """
    vocab = build_vocab_from_iterator(
        texts, min_freq=min_freq, specials=specials
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def collate_batch(
    batch: List[Tuple[List[str], int]], vocab: Vocab, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    將批次數據轉換為張量格式。
    
    Args:
        batch: 批次數據
        vocab: 詞彙表
        device: 設備
        
    Returns:
        (文本張量, 文本長度, 標籤張量)
    """
    text_list, labels = zip(*batch)
    text_lengths = torch.tensor([len(text) for text in text_list], dtype=torch.long)
    
    # 獲取最大文本長度
    max_length = max(text_lengths).item()
    
    # 將文本轉換為數字表示並填充
    padded_texts = []
    for text in text_list:
        tokens = [vocab[token] for token in text]
        padded = tokens + [vocab["<pad>"]] * (max_length - len(tokens))
        padded_texts.append(padded)
    
    # 轉換為張量
    text_tensor = torch.tensor(padded_texts, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    return text_tensor, text_lengths, labels_tensor


def create_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "review",
    label_col: str = "sentiment",
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader, Vocab]:
    """
    創建訓練和測試數據加載器。
    
    Args:
        train_df: 訓練數據
        test_df: 測試數據
        text_col: 文本列名
        label_col: 標籤列名
        batch_size: 批次大小
        device: 設備
        
    Returns:
        (訓練數據加載器, 測試數據加載器, 詞彙表)
    """
    # 對文本進行分詞
    train_texts = [tokenize_chinese(text) for text in train_df[text_col]]
    test_texts = [tokenize_chinese(text) for text in test_df[text_col]]
    
    # 從訓練集構建詞彙表
    vocab = build_vocabulary(train_texts)
    
    # 創建數據集
    train_labels = train_df[label_col].tolist()
    test_labels = test_df[label_col].tolist()
    
    train_dataset = ReviewDataset(train_texts, train_labels)
    test_dataset = ReviewDataset(test_texts, test_labels)
    
    # 創建數據加載器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, vocab, device),
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, vocab, device),
    )
    
    return train_dataloader, test_dataloader, vocab 