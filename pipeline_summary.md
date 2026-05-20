# NBME Score Clinical Patient Notes — Pipeline 說明

> **版本說明**：目前為 Baseline 版本（DeBERTa-v3-large），組員後續將在此基礎上做優化。

---

## 任務定義

給定一段**病歷（Patient Note）** 和一個**臨床特徵（Feature）**，找出病歷中哪些文字片段（Character Span）能佐證這個特徵。

```
Input:
  pn_history  : "HPI: 17yo M presents with palpitations.
                  ...dad with recent heart attcak..."
  feature_text: "Family history of MI"

Output:
  location    : "696 724"   ← 病歷中第 696~724 個字元
```

評估指標：**Character-level Micro F1**

---

## 整體流程總覽

```
原始資料 (CSV)
    │
    ▼
【資料前處理】preprocess_data.ipynb
  ├─ 合併三表 (train + features + patient_notes)
  ├─ 修正 40+ 筆錯誤 annotation
  ├─ 解析 location 字串 → char_spans (tuple list)
  ├─ feature_text 前處理（dash → 空白）
  └─ GroupKFold 5-fold 切分 (by pn_num)
       → 輸出 train_preprocessed_5fold.pkl
           test_preprocessed.pkl

    │
    ▼
【訓練】train_deberta_v3_large_colab.ipynb  (Google Colab Pro)
  ├─ 載入預訓練 DeBERTa-v3-large
  ├─ Tokenize: [CLS] pn_history [SEP] feature_text [SEP]
  ├─ 建立 token-level label (答案 span → 1, 其餘 → 0, special → -1)
  ├─ 5-fold 交叉訓練，每 fold 存最佳 F1 權重
  └─ 輸出：
       ├─ fold0~4 模型權重 (.pth)
       ├─ config.pth + tokenizer/
       └─ oof_df.pkl（全資料的驗證集預測，用於 threshold tuning）

    │
    ▼
【推論 & 提交】kaggle.ipynb  (Kaggle Notebook)
  ├─ 讀 oof_df.pkl → 搜尋最佳 threshold (0.45~0.55)
  ├─ 5-fold Ensemble（對 5 個模型的 char 機率取平均）
  ├─ 套用 threshold → 輸出 span 字串
  └─ 輸出 submission.csv → Kaggle 提交
```

---

## Notebook 對應說明

| Notebook | 平台 | 角色 |
|---|---|---|
| `preprocess_data.ipynb` | Colab / Local | 一次性資料前處理，輸出 pkl |
| `train_deberta_v3_large_colab.ipynb` | Google Colab Pro | 5-fold 訓練，輸出模型權重 |
| `kaggle.ipynb` | Kaggle Notebook | Threshold tuning + Ensemble 推論，輸出 submission |
| `train-deberta-v3-large-baseline.ipynb` | Kaggle Notebook | 原版 baseline 訓練（參考用） |
| `infer-deberta-v3-large-baseline.ipynb` | Kaggle Notebook | 原版 baseline 推論（參考用） |

---

## 模型選擇過程

| 階段 | 模型 | 結果 |
|---|---|---|
| 初版嘗試 | DeBERTa-v3-**base** (hidden_size 768) | Hidden state 訊噪比過低，token 分類鑑別力不足 |
| 最終版本 | DeBERTa-v3-**large** (hidden_size 1024) | 表示能力顯著提升，採用此版本提交 |

**Base → Large 的原因**：Base 模型在 token-level 二元分類任務中，預測機率分布過於平坦，無法清楚區分「屬於答案 span」和「不屬於答案 span」的 token。Large 的 hidden size 從 768 擴大到 1024（層數 12 → 24），representation 品質大幅提升。

---

## 模型架構

```
輸入：[CLS] pn_history tokens [SEP] feature_text tokens [SEP] [PAD...]
                                ↓
                  DeBERTa-v3-large Encoder
              (24 layers, hidden_size = 1024)
                                ↓
           Last Hidden States  [batch, seq_len, 1024]
                                ↓
                         Dropout(0.2)
                                ↓
                     Linear(1024 → 1)
                                ↓
             logits  [batch, seq_len, 1]
                                ↓
                    BCEWithLogitsLoss
         (只計算 pn_history token 的位置，忽略 special/padding/feature)
```

---

## 關鍵超參數

| 參數 | Baseline (原版 Kaggle) | 我們的版本 |
|---|---|---|
| 模型 | deberta-v3-large | deberta-v3-**base** → 切換為 deberta-v3-**large** |
| Batch size | 4 | 8 (× grad_accum 2 = 等效 16) |
| Epochs | 5 | 5 |
| Learning rate | encoder 2e-5 / decoder 2e-5 | 同左 |
| Scheduler | cosine | cosine |
| Max len | 512 | 動態計算（max 325） |
| fp16 (apex) | ✓ | ✗（PyTorch 2.6 相容性問題） |
| Folds trained | 1 (fold 0 only) | 5 (全部) |
| 平台 | Kaggle | Google Colab Pro + Kaggle |

---

## Token → Character 對應流程（推論核心）

```
token-level 機率  [0.02, 0.01, 0.87, 0.91, 0.88, 0.03, ...]
        ↓ offset_mapping（每個 token 對應的字元範圍）
char-level 機率   [0, 0, ..., 0.87, 0.87, 0.87, 0.91, ...]
        ↓ threshold（由 oof_df 調出最佳值，約 0.50）
binary mask       [0, 0, ..., 1, 1, 1, 1, ...]
        ↓ 找連續段落
span output       "203 217"
```

5-fold Ensemble 的做法：先對 5 個模型各別算出 char-level 機率，再取平均後套 threshold。

---

## Baseline 結果

| 指標 | 數值 |
|---|---|
| CV Score (5-fold OOF, th=0.51) | **0.85206** |
| Public LB Score | **0.85934** |
| Private Score | **0.86187** |

---

## 參考來源

### 主要參考：訓練架構
**yasufuminakama — NBME DeBERTa-v3-large Baseline Train**
- Kaggle Notebook：`train-deberta-v3-large-baseline.ipynb`
- 參考內容：整體模型架構（CustomModel）、train_fn / valid_fn / train_loop 結構、optimizer 設定（differential LR）、scoring functions、tokenizer 使用方式

我們在此基礎上做了以下修改：

| 修改項目 | 原版 | 我們的版本 |
|---|---|---|
| 資料讀取 | 讀 CSV + 手動修正 40+ 筆 annotation | 讀預處理好的 pkl（annotation 修正已內建） |
| `create_label` | 解析 location 字串格式 | 直接使用 char_spans tuple |
| fold 切分 | notebook 內做 GroupKFold | 已存入 pkl |
| fp16 | apex=True | apex=False（PyTorch 2.6 相容性問題） |
| batch size | 4 | 8（+ grad_accum 2，等效 16） |
| 訓練 fold 數 | 1（fold 0 only） | 5（全部） |
| wandb 記錄 | 可選 | 移除 |
| max_len | 固定 512 | 動態計算（max 325） |
| 平台 | Kaggle Notebook | Google Colab Pro |

---

## 後續優化方向（組員負責）

- [ ] 多模型 Ensemble（不同 seed / 不同模型）
- [ ] 更細緻的 threshold tuning（per feature / per case）
- [ ] 資料增強或 pseudo-labeling
- [ ] 學習率或訓練策略調整
- [ ] Post-processing（合併鄰近 span、去除噪音短 span）
