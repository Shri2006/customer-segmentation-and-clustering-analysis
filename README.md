# 📊 Customer Segmentation & Clustering Analysis

Applied unsupervised machine learning to segment customers into distinct, actionable groups for targeted marketing strategies.

![Dashboard Preview](charts/dashboard_preview.png)

**[🔗 View Live Interactive Dashboard](dashboard/index.html)** — the dashboard is a self-contained HTML file included in this repo (`dashboard/index.html`). To view it live on the web instead of downloading it, either:
- Enable **GitHub Pages** for this repo (Settings → Pages → deploy from `main` branch), then link to `https://<username>.github.io/<repo>/dashboard/`, or
- Open it instantly via [htmlpreview.github.io]([https://github.com/Shri2006/customer-segmentation-and-clustering-analysis/blob/main/index.html]) (swap in your GitHub username/repo)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Key Outcomes](#key-outcomes)
- [Challenges Overcome](#challenges-overcome)
- [Key Metrics](#key-metrics)
- [Data Visualizations](#data-visualizations)
- [Segment Profiles](#segment-profiles)
- [Key Insights & Recommendations](#key-insights--recommendations)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Project Overview

This project uses unsupervised machine learning to identify natural customer groupings from behavioral and demographic data. Rather than relying on arbitrary business rules ("customers who spent >$1000"), clustering algorithms let the data itself reveal which customers behave similarly — surfacing segments a human analyst might miss.

**K-Means clustering** was used as the primary model. To make sure the resulting segments weren't an artifact of one algorithm's assumptions, results were cross-validated against **Hierarchical (Agglomerative) clustering**, and the optimal number of clusters was selected using the **elbow method** and **silhouette analysis** rather than picked arbitrarily.

## Business Problem

Marketing spend is most effective when it's targeted. Treating all customers identically means high-value customers get generic offers (leaving money on the table) and at-risk customers get no attention until they've already churned. The goal of this project was to answer:

> *Can we group customers into a small number of segments — based only on their behavior and demographics — that are distinct enough to justify different marketing strategies for each?*

## Dataset

The dataset contains **1,000 customers** with 6 features:

| Feature | Description |
|---|---|
| `Age` | Customer age (years) |
| `Annual_Income_k` | Annual income (thousands, $) |
| `Spending_Score` | Behavioral spending score (1–100) |
| `Purchase_Frequency` | Orders placed per year |
| `Tenure_Months` | Months since customer acquisition |
| `Customer_Lifetime_Value` | Total historical revenue from customer ($) |

*Note: this dataset is synthetically generated (`data/generate_data.py`) to reflect realistic e-commerce customer distributions, since a proprietary customer dataset couldn't be shared publicly. The generation process embeds five latent behavioral groups without labeling them — the clustering pipeline has to rediscover that structure from the raw features alone, exactly as it would on a real dataset.*

## Methodology

1. **Exploratory Data Analysis** — distributions, correlations, and outlier checks across all 6 features.
2. **Feature Scaling** — standardized all features (`StandardScaler`) so that income (measured in thousands) doesn't dominate distance calculations over spending score (measured in single digits).
3. **Determining Optimal k** — ran K-Means for k = 2 through 10, plotting both:
   - **Elbow method** (inertia / within-cluster sum of squares)
   - **Silhouette score** (cluster cohesion vs. separation)
4. **K-Means Clustering** — fit the final model at k = 5.
5. **Hierarchical Clustering (validation)** — fit Ward-linkage agglomerative clustering independently at k = 5 and compared assignments to K-Means using the **Adjusted Rand Index**, to confirm the structure wasn't specific to one algorithm.
6. **Dimensionality Reduction (PCA)** — projected the 6-dimensional feature space down to 2 components for visualization, capturing 76.9% of total variance.
7. **Cluster Profiling** — computed mean feature values per cluster and translated statistical clusters into business-readable labels (e.g., "High-Value" instead of "Cluster 0").
8. **Validation** — since the synthetic data was generated with known ground-truth groups, the discovered segments were checked against them, confirming 94.5% recovery accuracy.

## Technologies Used

`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

- **Scikit-learn** — `KMeans`, `AgglomerativeClustering`, `StandardScaler`, `PCA`, `silhouette_score`, `adjusted_rand_score`
- **Pandas / NumPy** — data generation, wrangling, aggregation
- **Matplotlib / Seaborn** — all static visualizations (elbow curve, PCA scatter, heatmap, pie/bar charts)

## Key Outcomes

- ✅ Identified **5 distinct customer segments** with clear, interpretable characteristics
- ✅ K-Means and hierarchical clustering **agreed on structure** (Adjusted Rand Index: 0.845)
- ✅ Recovered the underlying segment structure with **94.5% accuracy** against validation labels
- ✅ Enabled a **personalized marketing and retention strategy per segment**
- ✅ Delivered results as both a reusable Python pipeline and a shareable interactive dashboard

## Challenges Overcome

- 🔷 **Determining the optimal number of clusters** — resolved by combining the elbow method with silhouette scoring rather than picking k arbitrarily
- 🔷 **Feature scaling and normalization** — income (tens of thousands) and spending score (1–100) needed standardizing before distance-based clustering would work correctly
- 🔷 **Validating cluster stability** — cross-checked K-Means against an entirely different algorithm (hierarchical) to make sure segments weren't an artifact of one method
- 🔷 **Translating statistical clusters into business labels** — mapped raw cluster IDs (0–4) to human-readable segment names using CLV, tenure, and spending patterns

## Key Metrics

| Metric | Value |
|---|---|
| Customers Analyzed | 1,000 |
| Customer Segments | 5 |
| K-Means Silhouette Score | 0.378 |
| Hierarchical Silhouette Score | 0.358 |
| Cluster Agreement (K-Means vs. Hierarchical, ARI) | 0.845 |
| PCA Variance Explained (2D) | 76.9% |
| Segment Recovery Accuracy (vs. ground truth) | 94.5% |
| Segment Coverage | 100% of customers |

## Data Visualizations

### Customer Distribution by Segment
![Segment Distribution](charts/segment_distribution_pie.png)

### Average Customer Lifetime Value by Segment
![CLV by Segment](charts/clv_by_segment_bar.png)

### Segment Feature Profile
![Segment Heatmap](charts/segment_heatmap.png)

### Cluster Separation (PCA Projection)
![PCA Clusters](charts/pca_clusters.png)

### Elbow Method & Silhouette Analysis
![Elbow and Silhouette](charts/elbow_silhouette.png)

## Segment Profiles

| Segment | % of Customers | Avg. Age | Avg. Income | Avg. Spending Score | Avg. Tenure (mo.) | Avg. CLV |
|---|---|---|---|---|---|---|
| **High-Value** | 15.2% | 43.0 | $103.5k | 84.4 | 47.1 | $5,307 |
| **Growth Potential** | 26.4% | 31.2 | $67.4k | 71.0 | 21.0 | $2,972 |
| **Loyal** | 28.2% | 47.0 | $55.0k | 55.5 | 60.9 | $2,236 |
| **At-Risk** | 16.9% | 38.9 | $60.6k | 27.3 | 31.2 | $1,338 |
| **New** | 13.3% | 27.1 | $50.8k | 38.8 | 6.3 | $785 |

## Key Insights & Recommendations

- **High-Value (15%)** — highest income, spending score, and CLV in the customer base. Prioritize retention with white-glove service, early access to new products, and loyalty perks; losing one of these customers costs far more than losing an average customer.
- **Growth Potential (26%)** — younger, high spending score but shorter tenure and lower income than High-Value. This is the best segment for **upsell and cross-sell campaigns** — they're already engaged, just haven't reached full spending potential yet.
- **Loyal (28%)** — the largest segment. Long tenure, steady mid-range spend. Best served with **loyalty programs and recognition** (e.g., tenure-based rewards) rather than acquisition-style discounts they don't need.
- **At-Risk (17%)** — moderate income but the lowest spending score and purchase frequency of any segment with meaningful tenure. This group needs **immediate re-engagement campaigns** (win-back offers, satisfaction surveys) before they churn entirely.
- **New (13%)** — shortest tenure, lowest CLV so far, but reasonable early spending score. Focus on **onboarding and early engagement** to convert them into the Growth Potential segment before they stagnate or churn.

## Project Structure

```
customer_segmentation/
├── README.md
├── data/
│   ├── generate_data.py          # synthetic dataset generator
│   └── customers_raw.csv         # raw customer data (pre-clustering)
├── notebooks/
│   └── customer_segmentation_analysis.py   # full clustering pipeline
├── outputs/
│   ├── customers_segmented.csv   # final labeled dataset
│   ├── segment_profile.csv       # per-segment feature averages
│   └── metrics.json              # model performance metrics
└── charts/                       # all generated visualizations + dashboard preview
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

python data/generate_data.py
python notebooks/customer_segmentation_analysis.py
```

Outputs (labeled dataset, segment profile, metrics, and all charts) are written to `outputs/` and `charts/`.
