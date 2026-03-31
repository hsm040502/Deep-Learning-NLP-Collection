# 基於 RNN 的情感分析模型 (RNN-based Sentiment Analysis Model)

本專案旨在開發一個基於循環神經網絡 (RNN) 的情感分析模型。該模型使用 PyTorch 框架進行建置，並以 `data/Comments_Mobile_Phone.xml` 資料集進行訓練與評估。最終產出一個命令列工具 (CLI)，能夠分析輸入文字的情感傾向，並將其分類為「正面」、「負面」或「中性」。

## 功能特色

*   從 XML 資料集中提取評論文字及其對應的真實情感標籤。
*   實作一個 LSTM (長短期記憶) 模型進行情感分析。
*   訓練模型以預測文字的情感分數 (0.0 至 1.0 之間)。
*   將預測的情感分數轉換為三分類標籤：正面 (1)、中性 (0)、負面 (-1)。
*   提供一個命令列介面，允許使用者分析 `data/Comments_Mobile_Phone.xml` 中的評論。
*   在命令列輸出每則評論的「真實情感分數」與「預測情感分數」。
*   計算並顯示整個模型在資料集上的「準確度 (Accuracy)」。

## 技術棧

*   **程式語言:** Python 3.10+
*   **深度學習框架:** PyTorch
*   **XML 剖析:** BeautifulSoup4
*   **數值運算:** NumPy
*   **文字處理:** NLTK (用於斷詞)
*   **評估指標:** scikit-learn (用於準確率計算)
*   **依賴管理與虛擬環境:** uv
*   **程式碼格式化與檢查:** Ruff
*   **測試框架:** pytest

## 目錄結構

```
SentimentAnalysis/
├── data/
│   └── Comments_Mobile_Phone.xml  # XML 評論資料檔案
├── src/                           # 核心原始碼
│   ├── __init__.py
│   ├── data_loader.py             # 資料載入與 XML 剖析
│   ├── preprocessing.py           # 文字預處理 (斷詞, 詞彙表, 序列化)
│   ├── model.py                   # PyTorch LSTM 模型定義
│   ├── trainer.py                 # 模型訓練、評估與預測邏輯
│   └── main.py                    # CLI 主執行腳本
├── tests/                         # 單元測試
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_trainer.py
│   └── test_main.py
├── spec/
│   └── PRD.md                     # 產品需求文件
├── pyproject.toml                 # 專案設定與依賴管理 (for uv & Ruff)
└── README.md                      # 本文件
```

## 環境設定

### 1. 前提條件

*   Python 3.10 或更高版本。
*   `uv` 已安裝。如果尚未安裝，請參考 [uv 官方文件](https://github.com/astral-sh/uv#installation) 進行安裝。

### 2. 建立虛擬環境與安裝依賴

在專案根目錄下執行以下指令：

```bash
# 1. 建立虛擬環境 (如果uv偵測到 pyproject.toml，可能會自動使用專案名稱)
uv venv

# 2. 啟用虛擬環境
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# 3. 安裝專案依賴
uv sync
# 或者，如果 uv sync 有問題，可以嘗試:
# uv pip install .
```

### 3. NLTK 資源

本專案使用 NLTK 進行文字斷詞，需要 `punkt` 資源。程式在首次執行 `TextPreprocessor` 時會嘗試自動下載。如果自動下載失敗，您可以手動下載：

```python
import nltk
nltk.download('punkt')
```
在 Python 直譯器中執行以上指令。

## 資料準備

將您的 XML 評論資料檔案 (預期名稱為 `Comments_Mobile_Phone.xml`) 放置在專案根目錄下的 `data/` 資料夾中。

如果您的 XML 檔案名稱或結構與 `src/data_loader.py` 中的預期不符，您可能需要修改該檔案中的解析邏輯。

## 如何執行程式

設定好環境並準備好資料後，您可以透過以下方式執行情感分析程式：

**方法一：直接執行 `main.py`**

```bash
python src/main.py [OPTIONS]
```

**方法二：使用 `pyproject.toml` 中定義的 script (推薦)**

```bash
analyze-sentiments [OPTIONS]
```
(請確保您的虛擬環境已啟用，並且 `uv sync` 或 `uv pip install .` 已成功將 script 安裝到環境中)

**可用選項 (OPTIONS):**

*   `--data_file TEXT`: XML 資料檔案的路徑 (預設: `data/Comments_Mobile_Phone.xml`)。
*   `--epochs INTEGER`: 訓練週期數 (預設: 10)。
*   `--batch_size INTEGER`: 訓練和預測的批次大小 (預設: 64)。
*   `--lr FLOAT`: 優化器的學習率 (預設: 0.001)。
*   `--help`: 顯示幫助訊息並退出。

**範例:**

```bash
analyze-sentiments --epochs 15 --batch_size 32
```

## 如何執行測試

本專案使用 `pytest` 進行單元測試。請確保已安裝開發依賴 (通常包含在 `uv sync` 中，或者您可以執行 `uv pip install ".[dev]"` 如果 `pyproject.toml` 有定義 `dev` extra)。

在專案根目錄下執行：

```bash
pytest
```

若要執行特定測試檔案：

```bash
pytest tests/test_data_loader.py
```

## 預期輸出格式 (CLI)

程式執行後，會針對資料集中的每則評論輸出以下資訊，最後顯示整體準確率：

```
INFO:src.main:
--- Sentiment Analysis Results ---

Review 1: 這手機的拍照功能真的超棒，夜拍也很清晰！...
  Raw Predicted Score: 0.9523
  True Sentiment:     1 (Positive)
  Predicted Sentiment: 1 (Positive)

Review 2: 電池續航力不太行，用一下就沒電了，有點失望。
  Raw Predicted Score: 0.1578
  True Sentiment:     -1 (Negative)
  Predicted Sentiment: -1 (Negative)

Review 3: 外觀還不錯，但系統偶爾會卡頓，整體普普通通。
  Raw Predicted Score: 0.4500
  True Sentiment:     1 (Positive) # 假設原始標籤是 Positive
  Predicted Sentiment: 0 (Neutral) # 預測為中性

...

INFO:src.main:
------------------------------------
INFO:src.main:Overall Model Accuracy on the dataset: XX.XX% (YY/ZZ)
INFO:src.main:------------------------------------
```
*(注意: 上述輸出中的真實情感標籤和預測結果僅為範例。)*
