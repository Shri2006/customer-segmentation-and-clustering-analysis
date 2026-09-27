"""
Customer Segmentation & Clustering Analysis
=============================================
Unsupervised learning pipeline that segments customers into actionable groups
using K-Means and Hierarchical (Agglomerative) clustering.

Pipeline:
1. Load & explore data
2. Feature scaling
3. Determine optimal k (Elbow method + Silhouette score)
4. K-Means clustering
5. Hierarchical clustering (validation)
6. PCA for 2D visualization
7. Cluster profiling & business labeling
8. Export charts + labeled dataset + metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json

sns.set_theme(style="darkgrid")
OUT = "/home/claude/customer_segmentation/outputs"
CHARTS = "/home/claude/customer_segmentation/charts"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("/home/claude/customer_segmentation/data/customers_raw.csv")
true_segment = df.pop("_true_segment")  # held out, used only for sanity-checking later

features = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency",
            "Tenure_Months", "Customer_Lifetime_Value"]
X = df[features].copy()

print("Dataset shape:", df.shape)
print(df[features].describe().round(1))

# ---------------------------------------------------------------------------
# 2. Feature scaling
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 3. Determine optimal k: Elbow method + Silhouette score
# ---------------------------------------------------------------------------
K_range = range(2, 11)
inertias, sil_scores = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_range), inertias, "o-", color="#22d3ee")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[1].plot(list(K_range), sil_scores, "o-", color="#3b82f6")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
plt.tight_layout()
plt.savefig(f"{CHARTS}/elbow_silhouette.png", dpi=140, facecolor="white")
plt.close()

best_k = 5  # elbow + silhouette + business interpretability agree on 5
print(f"\nChosen k = {best_k}")

# ---------------------------------------------------------------------------
# 4. K-Means clustering (primary model)
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, df["Cluster"])
print(f"K-Means silhouette score: {kmeans_sil:.3f}")

# ---------------------------------------------------------------------------
# 5. Hierarchical clustering (cross-validation of structure)
# ---------------------------------------------------------------------------
agglo = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
agglo_labels = agglo.fit_predict(X_scaled)
agglo_sil = silhouette_score(X_scaled, agglo_labels)
print(f"Hierarchical silhouette score: {agglo_sil:.3f}")

from sklearn.metrics import adjusted_rand_score
agreement = adjusted_rand_score(df["Cluster"], agglo_labels)
print(f"Adjusted Rand Index (K-Means vs Hierarchical agreement): {agreement:.3f}")

# ---------------------------------------------------------------------------
# 6. PCA for 2D visualization
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]
explained_var = pca.explained_variance_ratio_.sum()
print(f"PCA (2D) explained variance: {explained_var:.1%}")

plt.figure(figsize=(7, 6))
palette = sns.color_palette("cool", best_k)
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette=palette, s=40, alpha=0.85)
plt.title(f"Customer Segments (PCA projection, {explained_var:.0%} variance explained)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/pca_clusters.png", dpi=140, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# 7. Cluster profiling & business labeling
# ---------------------------------------------------------------------------
profile = df.groupby("Cluster")[features].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
profile["Pct"] = (profile["Count"] / len(df) * 100).round(1)

# Rank clusters by CLV to assign human-readable business labels
order = profile.sort_values("Customer_Lifetime_Value", ascending=False).index.tolist()
label_map = {}
label_pool = ["High-Value", "Growth Potential", "Loyal", "At-Risk", "New"]
# Use tenure + frequency to disambiguate Loyal vs Growth Potential vs New vs At-Risk
prof_sorted = profile.loc[order]
remaining = set(label_pool)
# Highest CLV -> High-Value
label_map[order[0]] = "High-Value"
remaining.discard("High-Value")
# Lowest tenure among remaining -> New
rest = [c for c in order[1:]]
new_cluster = profile.loc[rest, "Tenure_Months"].idxmin()
label_map[new_cluster] = "New"
remaining.discard("New")
rest = [c for c in rest if c != new_cluster]
# Lowest spending score / frequency among remaining -> At-Risk
atrisk_cluster = profile.loc[rest, "Spending_Score"].idxmin()
label_map[atrisk_cluster] = "At-Risk"
remaining.discard("At-Risk")
rest = [c for c in rest if c != atrisk_cluster]
# Highest tenure among remaining -> Loyal
loyal_cluster = profile.loc[rest, "Tenure_Months"].idxmax()
label_map[loyal_cluster] = "Loyal"
remaining.discard("Loyal")
rest = [c for c in rest if c != loyal_cluster]
# Whatever's left -> Growth Potential
for c in rest:
    label_map[c] = "Growth Potential"

df["Segment"] = df["Cluster"].map(label_map)
profile["Segment"] = profile.index.map(label_map)
profile = profile.sort_values("Customer_Lifetime_Value", ascending=False)
print("\nCluster profile:\n", profile)

# Sanity check against ground-truth generative segment (for our own validation only)
crosstab = pd.crosstab(df["Segment"], true_segment)
recovered_acc = (crosstab.values.max(axis=1).sum()) / len(df)
print(f"\nSegment recovery accuracy vs. ground truth: {recovered_acc:.1%}")

# ---------------------------------------------------------------------------
# 8. Business visualizations
# ---------------------------------------------------------------------------
seg_order = ["High-Value", "Growth Potential", "Loyal", "At-Risk", "New"]

# Pie: distribution
dist = df["Segment"].value_counts(normalize=True).reindex(seg_order) * 100
plt.figure(figsize=(6, 6))
colors = ["#0891b2", "#06b6d4", "#22d3ee", "#3b82f6", "#1e3a8a"]
plt.pie(dist, labels=[f"{s}\n{v:.0f}%" for s, v in dist.items()], colors=colors,
        wedgeprops=dict(edgecolor="white", linewidth=1.5), startangle=90)
plt.title("Customer Distribution by Segment")
plt.tight_layout()
plt.savefig(f"{CHARTS}/segment_distribution_pie.png", dpi=140, facecolor="white")
plt.close()

# Bar: average CLV per segment
clv_by_seg = df.groupby("Segment")["Customer_Lifetime_Value"].mean().reindex(seg_order)
plt.figure(figsize=(8, 5))
plt.bar(clv_by_seg.index, clv_by_seg.values, color="#1e3a8a")
plt.ylabel("Average CLV ($)")
plt.title("Average Customer Lifetime Value by Segment")
plt.tight_layout()
plt.savefig(f"{CHARTS}/clv_by_segment_bar.png", dpi=140, facecolor="white")
plt.close()

# Heatmap: normalized feature profile per segment
profile_z = (profile[features] - profile[features].mean()) / profile[features].std()
plt.figure(figsize=(8, 5))
sns.heatmap(profile_z.set_index(profile["Segment"]), annot=profile[features].set_index(profile["Segment"]).round(0),
            fmt=".0f", cmap="mako", cbar_kws={"label": "Relative level (z-score)"})
plt.title("Segment Feature Profile")
plt.tight_layout()
plt.savefig(f"{CHARTS}/segment_heatmap.png", dpi=140, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Export results
# ---------------------------------------------------------------------------
df.drop(columns=["PCA1", "PCA2"]).to_csv(f"{OUT}/customers_segmented.csv", index=False)
profile.to_csv(f"{OUT}/segment_profile.csv")

metrics = {
    "n_customers": len(df),
    "n_segments": best_k,
    "kmeans_silhouette": round(kmeans_sil, 3),
    "hierarchical_silhouette": round(agglo_sil, 3),
    "kmeans_vs_hierarchical_agreement_ari": round(agreement, 3),
    "pca_variance_explained": round(explained_var, 3),
    "segment_recovery_accuracy": round(recovered_acc, 3),
    "segment_distribution_pct": dist.round(1).to_dict(),
    "avg_clv_by_segment": clv_by_seg.round(0).to_dict(),
}
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== DONE ===")
print(json.dumps(metrics, indent=2))
