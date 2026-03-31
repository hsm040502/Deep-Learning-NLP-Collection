# Deep Learning & NLP Collection

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-green?style=flat)
![Jieba](https://img.shields.io/badge/Jieba-NLP-blue)

本倉庫收錄了我在深度學習 (Deep Learning) 與自然語言處理 (NLP) 領域的核心實作專案，涵蓋從基礎的神經網路建置、推薦系統設計，到進階的文本分析技術。

## 📂 專案目錄

1. [PyTorch 多類別分類器：玻璃成分等級預測](#1-pytorch-多類別分類器玻璃成分等級預測)
2. [ANN 聖誕送禮推薦系統 (含資料不平衡處理)](#2-ann-聖誕送禮推薦系統-含資料不平衡處理)
3. [NLP 文本前處理：中研院 Chinese Treebank 語料庫分析](#3-nlp-文本前處理中研院-chinese-treebank-語料庫分析)
4. [RNN 情緒分析模型 (Sentiment Analysis)](#4-rnn-情緒分析模型-sentiment-analysis)

---

## 1. PyTorch 多類別分類器：玻璃成分等級預測
**技術棧：** `PyTorch`, `Pandas`, `Scikit-learn`

### 📌 專案說明
利用 PyTorch 撰寫神經網路，針對 `Glass.csv` 資料集（含 214 筆玻璃成分資料）進行多類別分類。根據折射率 (RI) 及 8 種化學元素（Na, Mg, Al, Si, K, Ca, Ba, Fe）含量，將玻璃精準劃分為 7 個等級。

### 🛠 核心實作
- **資料前處理**：執行特徵標準化與類別標籤數位化，確保模型收斂。
- **模型架構**：建構多層感知器 (MLP) 處理多選一分類問題。
- **效能評估**：實現 `Y_test` 與 `Y_pred` 肩並肩輸出對比，並利用評估函式計算模型準確度。

---

## 2. ANN 聖誕送禮推薦系統
**技術棧：** `ANN`, `Keras/TensorFlow`, `Upsampling`, `One-Hot Encoding`

### 📌 專案說明
開發一個基於深度學習的推薦系統，解決聖誕節挑選禮物的繁瑣過程。用戶輸入受贈者的個人特徵（年齡、性別、個性、興趣等），系統自動推薦最適配的禮物類別。

### 🛠 核心實作
- **解決資料不平衡 (Imbalance Data)**：針對問卷收集到的偏態資料，使用 **上採樣 (Upsampling)** 技術調整類別分佈，提升小眾類別的預測準確性。
- **模型優化**：使用 `Nadam` 優化器與 `categorical_crossentropy` 損失函數，達成約 60% 的預測準確率（分類 5 大類別）。
- **特徵工程**：整合 9 項輸入變數，包含感性的「情感效果」與理性的「預算範圍」。

---

## 3. NLP 文本前處理：中研院 Chinese Treebank 語料庫分析
**技術棧：** `NLTK`, `Jieba`, `Regex`, `Named Entity Recognition (NER)`

### 📌 專案說明
深入解析中研院製作的 **Chinese Treebank (Sinica Treebank)** 語料庫，實作中文自然語言處理的核心流程。

### 🛠 核心實作
- **斷詞對比分析**：比對語料庫原始斷詞與 `Jieba` 斷詞的差異，深入理解中文斷詞之歧義性。
- **詞性標註 (POS Tagging)**：針對原始句子進行自動化詞性標註，並與基準 (Ground Truth) 進行並列校對。
- **實體命名辨識 (NER)**：撰寫辨識邏輯提取句子中的實體名稱（人名、地名、組織名等），並過濾不含實體之句型，提升資料純度。

---

## 4. RNN 情緒分析模型 (Sentiment Analysis)
**技術棧：** `RNN (Recurrent Neural Networks)`, `Embedding`, `NLP`

### 📌 專案說明
針對 3C 產品評論進行情感極性分析。詳細內容與程式碼實作請參閱子資料夾內之單獨 README：
👉 [查看完整專案細節](./SentimentAnalysis)

---

## 🚀 學習成效總結
透過這些專案，我掌握了：
- **深度學習框架**：能靈活運用 PyTorch 與 TensorFlow 解決分類與推薦問題。
- **資料清理技術**：處理過標準化 (Standardization)、標籤編碼 (One-Hot) 與資料平衡 (Sampling) 等關鍵步驟。
- **中文 NLP 領域**：具備從原始語料庫到斷詞、詞性標註、NER 的全流程實作能力。
