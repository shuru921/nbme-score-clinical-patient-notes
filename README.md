# NBME - Score Clinical Patient Notes (DeBERTa-v3)

Kaggle 競賽 [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes) 的訓練與推論程式碼，使用 `microsoft/deberta-v3-base` 和 `microsoft/deberta-v3-large`。

## 任務說明

從病人病歷（`pn_history`）中找出對應臨床特徵（`feature_text`）的文字片段（span extraction）。

- **輸入**：病歷文字 + 臨床特徵名稱
- **輸出**：病歷中對應特徵的字元位置（`start end`）
- **評估指標**：Character-level Micro F1

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `train_deberta_v3_base_colab.ipynb` | DeBERTa-v3-base 訓練（Google Colab） |
| `train_deberta_v3_large_colab.ipynb` | DeBERTa-v3-large 訓練（Google Colab Pro，建議 A100） |
| `infer-deberta-v3-base-colab.ipynb` | DeBERTa-v3-base 推論（Colab） |
| `notebook770d150c35.ipynb` | DeBERTa-v3-large 推論，輸出 `submission.csv`（Kaggle 環境，實際提交版） |
| `pipeline_summary.md` | 完整 pipeline 說明文件 |
| `ref/` | 參考用 baseline notebook（librauee） |

## 模型架構

```
[CLS] pn_history_tokens [SEP] feature_text_tokens [SEP]
                    ↓
             DeBERTa backbone
                    ↓
         last hidden states [batch, seq_len, hidden_size]
                    ↓
               Dropout(0.2)
                    ↓
           Linear(hidden_size → 1)
                    ↓
          token-level logits → sigmoid → char-level 機率 → span
```

Loss 使用 `BCEWithLogitsLoss`，只計算 pn_history 部分的 token（feature_text 和特殊 token 的 label 設為 -1 忽略）。

## 訓練設定

| 超參數 | base | large |
|--------|------|-------|
| Epochs | 5 | 5 |
| Batch size | 16 | 8（grad accum ×2，等效 16）|
| Encoder LR | 2e-5 | 2e-5 |
| Scheduler | Cosine | Cosine |
| Max length | 325（動態計算）| 325（動態計算）|
| Folds | 5-fold CV | 5-fold CV |
| fp16 | ✗（PyTorch 2.6 相容性問題）| ✗ |

使用 differential LR（backbone 和分類頭分開設定）。

## 結果

| 指標 | 數值 |
|------|------|
| CV Score (5-fold OOF, th=0.51) | **0.852** |
| Public LB | **0.859** |
| Private Score | **0.862** |

## 資料來源

前處理後的資料由組員整理，來源：[yuyu-819/NBME_Score-Clinical-Patient-Notes](https://github.com/yuyu-819/NBME_Score-Clinical-Patient-Notes)

使用的檔案：
- `train_preprocessed_5fold.pkl`：訓練資料，已合併三表、解析 char_spans、完成 5-fold 切分（共 14,300 筆）
- `test_preprocessed.pkl`：測試資料

## 資料流程

```
train_preprocessed_5fold.pkl  ← 來自 yuyu-819/NBME_Score-Clinical-Patient-Notes
        ↓
  feature_text 前處理（dash → 空白）
        ↓
  DebertaV2TokenizerFast tokenize
        ↓
  建立 token-level label（依 char_spans）
        ↓
  5-fold 訓練 → 每 epoch 計算 char-level F1 → 存最佳模型
        ↓
  oof_df.pkl（供 inference threshold tuning）
        ↓
  Kaggle inference → submission.csv
```

## 使用方式

### 訓練（Colab）

1. 從 [yuyu-819/NBME_Score-Clinical-Patient-Notes](https://github.com/yuyu-819/NBME_Score-Clinical-Patient-Notes) 取得前處理好的資料，上傳至 Google Drive：
   ```
   MyDrive/NBME_Score-Clinical-Patient-Notes/data/.../processed/
   ├── train_preprocessed_5fold.pkl
   └── test_preprocessed.pkl
   ```
2. 開啟對應的訓練 notebook，修改 `BASE_DIR` / `OUTPUT_DIR` 路徑後執行。
3. 輸出的模型權重和 `oof_df.pkl` 會存到 `OUTPUT_DIR`。

### 推論（Kaggle）

1. 將訓練輸出（`config.pth`、`tokenizer/`、`oof_df.pkl`、5 個 fold 的 `.pth`）上傳為 Kaggle Dataset。
2. 在 `notebook770d150c35.ipynb` 中修改 `MODEL_DIR` 指向該 Dataset。
3. 執行 notebook，輸出 `/kaggle/working/submission.csv`。

## 參考

- 訓練 baseline：[librauee/train-deberta-v3-large-baseline](https://www.kaggle.com/code/librauee/train-deberta-v3-large-baseline)
- 推論 baseline：[librauee/infer-deberta-v3-large-baseline](https://www.kaggle.com/code/librauee/infer-deberta-v3-large-baseline)
- 模型：[microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)、[microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
