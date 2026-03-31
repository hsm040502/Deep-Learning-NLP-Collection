"""
3C產品評論情緒分析示例

本腳本演示如何使用RNN模型對3C產品評論進行情緒分析。
"""
import sys
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 確保能夠導入src模塊
sys.path.append(os.path.abspath('..'))

from src.data import load_data, create_dataloaders, tokenize_chinese
from src.models import RNNSentimentClassifier
from src.train import train, evaluate_model, predict

# 設置中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 加載數據
    print("## 1. 加載數據")
    data_path = '../data/sample_data.csv'
    train_df, test_df = load_data(data_path)

    print(f"訓練集大小: {len(train_df)}")
    print(f"測試集大小: {len(test_df)}")
    print("\n訓練集前幾條數據:")
    print(train_df.head())

    # 2. 數據分析與可視化
    print("\n## 2. 數據分析與可視化")
    
    # 分析情緒標籤分佈
    sentiment_counts = train_df['sentiment'].value_counts().sort_index()
    
    # 加載標籤映射
    with open('../configs/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    print("\n情緒標籤分佈:")
    for idx, count in sentiment_counts.items():
        print(f"{label_map[str(idx)]}: {count}")
    
    # 分析評論長度
    train_df['length'] = train_df['review'].apply(lambda x: len(tokenize_chinese(x)))
    print(f"\n評論平均長度: {train_df['length'].mean():.2f}")
    print(f"評論最大長度: {train_df['length'].max()}")
    print(f"評論最小長度: {train_df['length'].min()}")
    
    # 3. 準備數據加載器
    print("\n## 3. 準備數據加載器")
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建數據加載器
    batch_size = 8  # 小批次大小，用於演示
    train_dataloader, test_dataloader, vocab = create_dataloaders(
        train_df, test_df, batch_size=batch_size, device=device
    )
    
    print(f"詞彙表大小: {len(vocab)}")
    
    # 4. 創建與訓練模型
    print("\n## 4. 創建與訓練模型")
    
    # 加載模型配置
    with open('../configs/model_config.json', 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    
    # 創建模型
    model = RNNSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        n_layers=model_config['n_layers'],
        bidirectional=model_config['bidirectional'],
        dropout=model_config['dropout'],
        rnn_type=model_config['rnn_type'],
        pad_idx=vocab['<pad>'],
    ).to(device)
    
    # 顯示模型信息
    print("模型結構:")
    print(model)
    
    # 設定優化器和損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 訓練模型（此處僅訓練少量周期以供演示）
    n_epochs = 3
    model_save_path = '../models/demo_model.pt'
    vocab_save_path = '../models/vocab.pt'
    
    # 保存詞彙表
    os.makedirs('../models', exist_ok=True)
    torch.save(vocab, vocab_save_path)
    print(f"詞彙表已保存到: {vocab_save_path}")
    
    # 訓練模型
    print("\n開始訓練模型...")
    history = train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        n_epochs,
        device,
        model_save_path,
        patience=3,
    )
    
    # 5. 模型預測測試
    print("\n## 5. 模型預測測試")
    
    # 加載最佳模型
    model.load_state_dict(torch.load(model_save_path))
    
    # 準備一些測試文本
    test_texts = [
        "這款手機非常好用，續航持久，攝像頭也很清晰。",
        "這個藍牙耳機音質還不錯，就是有時候會斷連。",
        "這台電腦太差了，又卡又慢，而且價格也不便宜。",
    ]
    
    print("測試文本預測結果:")
    # 預測情緒
    predictions = predict(model, test_texts, vocab, tokenize_chinese, device)
    
    # 輸出預測結果
    for text, pred in zip(test_texts, predictions):
        print(f"文本: {text}")
        print(f"情緒: {label_map[str(pred)]}")
        print("---")

if __name__ == "__main__":
    main() 