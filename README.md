# 📈 Bitcoin Price Prediction Project

**By Anoop Bhaskar & Subin Pradeep**

---

## 🧠 What Is This Project?

This project attempts to **predict the daily market price of Bitcoin** using historical blockchain data and machine learning models. Bitcoin is notoriously volatile and hard to forecast — making it an interesting and challenging prediction problem.

We used data spanning **2010 to 2018**, applied careful data cleaning and feature engineering, and tested three different regression models to see which one could most accurately predict Bitcoin's price — both for the current day and several days into the future.

---

## 📦 The Dataset

The dataset was inspired by Kaggle and contains blockchain and market metrics such as:
- Number of unique Bitcoin addresses
- Transaction volume (in USD)
- Hash rate (a measure of mining activity)
- Transaction fees
- Block size and block count
- Trade volume, and more

> Think of these as the "vital signs" of the Bitcoin network — the numbers that describe how actively Bitcoin is being mined, traded, and used.

---

## 🧹 How We Cleaned and Prepared the Data

Raw data is messy. Before building any model, we went through a thorough data preparation process:

### 1. 🗑️ Removed Zero-Price Entries
Early Bitcoin records had market prices of $0 (before it had real value). We removed those to avoid misleading the model.

### 2. 🩹 Filled in Missing Values
One variable (`btc_trade_volume`) had 21 missing entries. We used **spline interpolation** — a technique that draws a smooth curve through existing data points to estimate what the missing values should be. This works especially well for financial time-series data.

### 3. 🚨 Flagged Outliers (Without Removing Them)
Bitcoin's 2017 price surge is dramatic — but it's real, not an error. Instead of deleting these extreme values, we **flagged them** using a 30-day rolling window to track when any feature behaved unusually. This preserved historical integrity while giving the model a heads-up.

### 4. 📐 Log Transformation
Many financial variables are skewed (e.g., a $500 change means very different things at $600 vs. $15,000). We applied a **log transformation** to make these percentage-based changes more linear and easier for models to learn.
> The model ultimately predicts `log(bitcoin_price)`, which is standard in financial forecasting.

### 5. ⚖️ Standardization
We scaled all features so that no single variable dominates just because it has larger numbers. This ensures a fair comparison across all inputs.

### 6. 🔗 Removed Redundant Features (Multicollinearity)
Many blockchain metrics move together (e.g., more transactions → more fees). Having too many correlated features confuses regression models. We used **VIF scores** (Variance Inflation Factor) to identify and remove the most redundant variables, ending up with 12 key features.

### 7. ⏳ Added Lagged Features & Rolling Averages
Since Bitcoin's price today is influenced by what happened recently, we engineered:
- **Lagged features**: past values of each variable (e.g., yesterday's price, 3 days ago, etc.)
- **7-day rolling averages**: the trend over the past week
- **7-day rolling standard deviations**: how volatile things have been recently

This gives the model both a memory of the recent past and a sense of current trends.

### 8. ✂️ Train/Test Split
Because this is time-series data, we **cannot** split randomly (that would be like using tomorrow's news to predict yesterday's price). Instead, we used:
- **Training set**: 2010 → mid-2016 (first 80%)
- **Testing set**: mid-2016 → 2018 (last 20%)

---

## 🤖 The Models We Used

We tested three regression models of increasing sophistication:

### 1. Multiple Linear Regression (MLR) — The Baseline
The simplest model. It finds the best straight-line relationship between features and price. Easy to interpret, but it struggles when features are correlated with each other (which ours were).

### 2. Ridge Regression — Regularized
Adds a penalty that **shrinks all feature weights** toward zero, which helps stabilize predictions when features are correlated. It keeps all features but reduces the influence of noisy ones.

### 3. Lasso Regression — Feature-Selecting
Similar to Ridge, but it can **eliminate features entirely** by setting their weights to exactly zero. This is powerful because it automatically identifies which variables are most useful, ignoring the rest.

---

## 📅 Forecasting Horizons

We didn't just predict today's price — we tested each model's ability to look ahead:

| Horizon | Meaning |
|--------|---------|
| **t** | Current day |
| **t+1** | 1 day into the future |
| **t+3** | 3 days into the future |
| **t+7** | 7 days into the future |

This reveals how well each model holds up as we ask it to predict further ahead.

---

## 📊 Results

### Current-Day Predictions (Horizon: t)

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | 0.0857 | 0.0579 | 0.993 |
| Ridge Regression | 0.0578 | 0.0401 | 0.996 |
| **Lasso Regression** | **0.0461** | **0.0303** | **0.998** |

> **R²** measures how much of the price variation is explained by the model (1.0 = perfect). **RMSE** and **MAE** measure average error (lower = better).

All three models performed excellently for same-day predictions. But the real test was predicting the *future*.

---

### Forecasting Performance Across Horizons

| Horizon | Model | RMSE | R² |
|--------|-------|------|----|
| t+1 | MLR | 0.2216 | 0.955 |
| t+1 | Ridge | 0.1589 | 0.977 |
| t+1 | **Lasso** | **0.0653** | **0.996** |
| t+3 | MLR | 0.4661 | 0.880 |
| t+3 | Ridge | 0.3634 | 0.879 |
| t+3 | **Lasso** | **0.0951** | **0.992** |
| t+7 | MLR | 0.9559 | 0.161 |
| t+7 | Ridge | 0.7372 | 0.501 |
| t+7 | **Lasso** | **0.1369** | **0.983** |

### Key Takeaways:
- **Lasso dominated at every horizon** — its ability to drop irrelevant features kept it accurate even 7 days out.
- **MLR collapsed at t+7** — an R² of just 0.16 means it explained almost none of the variance a week ahead.
- **Ridge was better than MLR** but still held onto noisy features that hurt its long-range predictions.
- **All models struggled more as the forecast horizon increased** — exactly what we expected given Bitcoin's volatility.

---

## 💡 Conclusions

This project showed that:

1. **Data preparation is half the battle.** Cleaning, transforming, and engineering features made a massive difference in model quality.
2. **Regularization matters — especially Lasso.** When data has many correlated and noisy features, automatically selecting the most useful ones leads to much better predictions.
3. **Bitcoin is hard to predict far in advance.** Even the best model degrades as you look further into the future — which is expected for such a volatile asset.
4. **Simpler models break down under complexity.** Standard linear regression, while a useful baseline, is poorly suited for volatile, multicollinear financial data.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Models**: `LinearRegression`, `Ridge`, `Lasso` (from scikit-learn)
- **Preprocessing**: `StandardScaler`, spline interpolation, VIF analysis
