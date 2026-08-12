# Customer Segmentation via PCA — UK Online Retail

A portfolio project applying Principal Component Analysis (PCA) to reduce dimensionality of customer behavioral data, followed by K-Means clustering to segment customers into actionable groups for a UK-based online retailer.

## Overview

Retailers often have many overlapping behavioral signals about their customers (spend, frequency, basket size, product variety, tenure, etc.). This project demonstrates how PCA can compress correlated behavioral variables into a smaller set of interpretable components — and shows, with measurable evidence, that doing so improves clustering quality rather than just simplifying the data for its own sake.

**Key result:** PCA-reduced features (4 components) consistently outperformed raw scaled features on silhouette score across every tested cluster count, while reducing dimensionality from 7 behavioral features down to 4.

## Dataset

[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — transaction-level data from a UK-based online retailer, December 2010 to December 2011 (~541,909 transactions).

Scoped to **UK-only customers** (~90% of transactions) to maintain behavioral coherence and avoid sparse, noisy signal from the ~30+ low-volume countries in the full dataset.

## Project Structure

```
├── PCA_Customer_Segmentation.ipynb   # Full analysis notebook (all phases)
├── online_retail.csv                 # Raw dataset (converted from .xlsx)
├── online_retail_cleaned.csv         # Post-cleaning, UK-only transactions
├── customer_features.csv             # Customer-level RFM + behavioral features
├── customer_features_transformed.csv # + log-transformed features
├── scaled_features.csv               # Standardized feature set (pre-PCA)
├── pca_features_v2.csv               # Final 4-component PCA output
├── pca_loadings_v2.csv               # Feature loadings per component
├── customer_clusters.csv             # Final cluster assignments
└── README.md
```

## Methodology

### 1. Data Cleaning
Dropped transactions with missing `CustomerID` (~25% of rows), removed cancelled orders (invoice numbers prefixed `C`), removed zero/negative unit prices, dropped duplicates, and filtered to UK-only customers. Resulted in **3,920 unique customers**.

### 2. Feature Engineering
Built a customer-level table combining classic RFM metrics with additional behavioral features:

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of distinct orders |
| Monetary | Total amount spent |
| AvgBasketSize | Average items per order |
| AvgOrderValue | Average spend per order |
| ProductVariety | Number of distinct products purchased |
| Tenure | Days since first purchase |
| OrderTrend | Change in order volume, early vs. late half of the observed period |

### 3. Exploratory Analysis
Correlation analysis justified PCA before applying it — notable correlations included `AvgBasketSize`/`AvgOrderValue` (0.93), `Frequency`/`ProductVariety` (0.65), and `Monetary`/`Frequency` (0.51). Most monetary/frequency-related features were heavily right-skewed.

### 4. Preprocessing
Applied `log1p` transformation to skewed features, then standardized all features (mean ≈ 0, std ≈ 1) using `StandardScaler`.

An initial PCA pass included one-hot encoded `FavDayOfWeek`, but this caused 4 of 8 components to be driven almost entirely by day-of-week dummy variables with little behavioral signal mixed in. This feature was excluded from the final model — a deliberate finding documented in the notebook, not a silent fix.

### 5. PCA
Fit on the 7 remaining continuous behavioral features. **4 principal components retained 90.11% of total variance**, each cleanly interpretable:

- **PC1 — Spending Intensity:** driven by Monetary, Frequency, ProductVariety, AvgOrderValue
- **PC2 — Tenure / Lifecycle:** driven by Tenure, Recency, OrderTrend
- **PC3 — Purchase Style:** bulk buying (AvgOrderValue, AvgBasketSize) vs. frequent smaller purchases
- **PC4 — Order Trend / Recency Shift:** driven by OrderTrend and Recency

### 6. Clustering — PCA vs. Raw Features
Ran K-Means with `k = 2` through `10` on both the raw scaled behavioral features and the 4-component PCA output, comparing silhouette scores at each `k`. PCA-based clustering scored higher at every value of `k` tested (e.g., at k=3: PCA ≈ 0.29 vs. raw ≈ 0.23). Final model: **k = 3**, chosen from a combination of the elbow method and silhouette score.

High-value outlier customers (identified during EDA) were deliberately retained rather than removed — post-clustering analysis confirmed they were coherently grouped within the high-value segment rather than distorting cluster boundaries, validating that decision.

### 7. Cluster Profiles

| Segment | Size | Profile | Suggested Action |
|---|---|---|---|
| **Champions** | 38.1% | High spend, frequent, recent, long tenure | Retention: loyalty perks, early access |
| **Emerging** | 35.6% | Newer customers, lower volume, positive order trend | Growth: onboarding nudges, cross-sell |
| **At-Risk** | 26.3% | Long tenure, high recency (dormant), negative order trend | Win-back: reactivation campaigns |

## Tools & Libraries

- Python, Jupyter Notebook
- pandas, numpy
- scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)
- matplotlib, seaborn

## Limitations

- Single-country (UK) scope; findings may not generalize to other markets
- One year of data; no seasonality analysis
- K-Means assumes roughly spherical clusters, which may not capture all real-world segment shapes
- `FavDayOfWeek` was explored but ultimately excluded — a different encoding strategy might recover some of that signal

## How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `PCA_Customer_Segmentation.ipynb` in Jupyter and run all cells in order

---

*This project was built as a learning and portfolio exercise in dimensionality reduction and unsupervised customer segmentation.*
