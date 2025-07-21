# Mirai22K - Gold Futures Directional Movement Classifier

This repository contains a machine learning model trained to predict **directional price movement** of **U.S. Gold Futures (COMEX: GC=F)**. It leverages technical and macroeconomic indicators, with emphasis on feature engineering and deployment readiness.

> ⚠️ **Disclaimer**: This tool is developed strictly for **educational and research purposes**. It is **not financial advice**. Do not use it for real-world trading without professional validation, risk controls, and regulatory compliance.

---

## Objective

To build a supervised binary classifier that outputs whether the **next trading session** will experience a **price increase (`1`)** or **decrease (`0`)** in gold futures, along with the **confidence level** of that prediction.

---

## Model Overview

| Element            | Description |
|--------------------|-------------|
| **Type**           | `RandomForestClassifier` (sklearn) |
| **Output**         | Binary class + probability |
| **Format**         | `.pkl` via `joblib` |
| **Deployment**     | Colab/CLI/Script-ready |
| **Target**         | Daily delta in gold futures closing price |

---

## Methodology & Pipeline

### 1. **Data Collection**
- Source: `yfinance`
- Tickers: `"GC=F"` (Gold Futures), `"DXY"` (Dollar Index), `"^TNX"` (US 10-Year Treasury)
- Date Range: `2020-01-01` to `datetime.today()`

### 2. **Raw Features**
| Ticker | Extracted Field | Description |
|--------|------------------|-------------|
| GC=F   | `Close`, `Volume` | Spot price of gold, COMEX trading volume |
| DXY    | `Close`          | U.S. dollar strength proxy |
| TNX    | `Close`          | Treasury yield (interest rate proxy) |

### 3. **Feature Engineering**
- Log Returns (`ret_1`, `ret_5`)
- Differenced momentum (`tnx_diff`, `dxy_ret`)
- Lag variables: `dxy_lag1`, `tnx_lag1`, `dxy_lag2`, ..., `tnx_lag5`
- Derived COMEX volume (not used in final model)
- Final feature shape: **13 features**

### 4. **Labeling Strategy**
```python
df['Target'] = (df['Spot_Price'].shift(-1) > df['Spot_Price']).astype(int)
```
The target is `1` if next-day's gold price closes higher, else `0`.

### 5. **Model Training**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)
```

### 6. **Evaluation**
- Accuracy: ~87%
- AUC Score: ~0.91
- Tools: `classification_report`, `roc_curve`, `confusion_matrix`
- Manual test case inference confirmed realistic outputs

---

## Feature List

```text
['Spot_Price', 'DXY', 'TNX', 'ret_1', 'ret_5', 'dxy_ret',
 'tnx_diff', 'dxy_lag1', 'tnx_lag1', 'dxy_lag2', 'tnx_lag2',
 'dxy_lag5', 'tnx_lag5']
```

---

## Files in Repository

```
.
├── gold_prob_model.pkl        # Trained and serialized model
├── Mirai22K.ipynb             # Colab notebook for loading and testing the model
├── README.md                  # This documentation
├── LICENSE                    # Apache 2.0 license
```

---

## Libraries Used

| Library         | Purpose |
|-----------------|---------|
| `pandas`, `numpy` | Data manipulation |
| `yfinance`        | Historical market data |
| `sklearn`         | Model training & evaluation |
| `joblib`          | Model serialization |
| `matplotlib`, `seaborn` | Visualization |
| `plotly`          | Interactive charts (optional) |

---

## License

Distributed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## Contributions

This is a single-developer research effort for demonstrating market ML workflows. For collaborations, improvements, or extensions (e.g., LSTM, XGBoost, live-feed backtesting), feel free to open a PR or start a discussion.

---

**This project does not give trading signals. Use it only as a research tool.**
