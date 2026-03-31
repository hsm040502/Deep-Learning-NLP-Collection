"""
RNN模型定義模塊。
"""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class RNNSentimentClassifier(nn.Module):
    """
    使用RNN進行情緒分析的模型。
    
    Attributes:
        embedding: 詞嵌入層
        rnn: RNN層 (可以是LSTM或GRU)
        fc: 全連接層
        dropout: Dropout層
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5,
        rnn_type: str = "lstm",
        pad_idx: Optional[int] = None,
    ) -> None:
        """
        初始化RNN情緒分析模型。
        
        Args:
            vocab_size: 詞彙表大小
            embedding_dim: 詞嵌入維度
            hidden_dim: 隱藏層維度
            output_dim: 輸出維度（情緒類別數量）
            n_layers: RNN層數
            bidirectional: 是否使用雙向RNN
            dropout: Dropout比率
            rnn_type: RNN類型，可選 "lstm" 或 "gru"
            pad_idx: 填充索引，用於在嵌入層中屏蔽填充標記
        """
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        
        # 選擇RNN類型
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        
        self.rnn = rnn_cls(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        # 如果是雙向RNN，則隱藏層維度乘2
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, text: Tensor, text_lengths: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向傳播。
        
        Args:
            text: 輸入文本序列，形狀為 [batch_size, seq_len]
            text_lengths: 每個序列的實際長度，形狀為 [batch_size]
            
        Returns:
            tuple: (預測結果, 隱藏狀態)
        """
        # text = [batch size, seq len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, embedding dim]
        
        # 打包序列以忽略填充標記
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        if isinstance(self.rnn, nn.LSTM):
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:  # GRU
            packed_output, hidden = self.rnn(packed_embedded)
        
        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output = [batch size, seq len, hidden dim * num directions]
        
        # 如果是雙向RNN，則合併前向和後向的最後隱藏狀態
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=1))
        else:
            hidden = self.dropout(hidden[-1])
        # hidden = [batch size, hidden dim]
        
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        
        return prediction, hidden
        
    def load_pretrained_embeddings(self, embeddings: Tensor) -> None:
        """
        加載預訓練的詞嵌入。
        
        Args:
            embeddings: 預訓練的詞嵌入矩陣
        """
        self.embedding.weight.data.copy_(embeddings) 