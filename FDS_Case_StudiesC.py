"""
FDS Case Studies: Customer Segmentation and Wine Quality Assessment
Standalone Python script generated from the updated notebook.

Expected files:
- EastWestAirlines.xlsx (sheet: data)
- winequality-red.csv
- winequality-white.csv

Place datasets in the project root or inside a folder named "data" or "sample_data".
Run from the project root: python FDS_Case_Studies.py
"""

from pathlib import Path
import os
import warnings
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

def find_data_file(filename):
    """Find a dataset in the project root or common data subfolders."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "sample_data" / filename,
        BASE_DIR / "datasets" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    tried = "\\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Put it in the project root or a data/ folder. Tried:\\n{tried}"
    )


# ===== Notebook code cell 1 =====
# Purpose: import all libraries used in both case studies and set a common plot style
import os, warnings
from IPython.display import display
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
sns.set_theme(style='whitegrid')
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)

# ===== Notebook code cell 6 =====
# Purpose: load the airline data and check its size, missing values and duplicate rows
air = pd.read_excel(find_data_file('EastWestAirlines.xlsx'), sheet_name='data')
print('Shape:', air.shape)
print('Missing values:', air.isna().sum().sum())
print('Duplicate rows:', air.duplicated().sum())
air.head()

# ===== Notebook code cell 9 =====
# Purpose: summarise the numeric features, remove outliers (|z| > 3) and standardise the data
feats = ['Balance','Qual_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Days_since_enroll']
print(air[feats].describe().T.round(1))
print('\nSkewness of each feature:\n', air[feats].skew().round(2))

zs = np.abs((air[feats] - air[feats].mean()) / air[feats].std())
print('\nRows with at least one |z| > 3:', (zs > 3).any(axis=1).sum())
air_c = air[(zs <= 3).all(axis=1)].reset_index(drop=True)
print('Rows kept after removing outliers:', len(air_c), f'({len(air_c)/len(air):.1%})')
Xa = StandardScaler().fit_transform(air_c[feats])

# ===== Notebook code cell 12 =====
# Purpose: draw a histogram for each clustering feature
fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
for ax, c in zip(axes, feats):
    sns.histplot(air_c[c], bins=30, color='steelblue', ax=ax)
    ax.set_title(f'Distribution of {c}'); ax.set_xlabel(c); ax.set_ylabel('Number of members')
axes[-1].axis('off')
plt.tight_layout(); plt.savefig('figures/cs2_histograms.png', dpi=130); plt.show()

# ===== Notebook code cell 14 =====
# Purpose: draw a correlation heatmap of the features and the Award flag
plt.figure(figsize=(8, 6.5))
sns.heatmap(air_c[feats + ['Award?']].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=.5, cbar_kws={'label': 'Pearson correlation'})
plt.title('Correlation Heatmap of Clustering Features and Award Flag')
plt.tight_layout(); plt.savefig('figures/cs2_correlation.png', dpi=130); plt.show()

# ===== Notebook code cell 16 =====
# Purpose: scatter plots and a box plot to look for hidden patterns
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
sns.scatterplot(data=air_c, x='Bonus_miles', y='Balance', alpha=.4, color='steelblue', ax=axes[0])
axes[0].set_title('Balance vs Bonus Miles'); axes[0].set_xlabel('Bonus miles'); axes[0].set_ylabel('Balance (miles)')
sns.scatterplot(data=air_c, x='Flight_trans_12', y='Flight_miles_12mo', alpha=.4, color='steelblue', ax=axes[1])
axes[1].set_title('Flight Miles vs Flight Transactions (12 months)'); axes[1].set_xlabel('Flight transactions'); axes[1].set_ylabel('Flight miles')
sns.boxplot(data=air_c, x='Award?', y='Bonus_miles', color='lightgray', ax=axes[2])
axes[2].set_title('Bonus Miles by Award Flag'); axes[2].set_xlabel('Award (0 = no, 1 = yes)'); axes[2].set_ylabel('Bonus miles')
plt.tight_layout(); plt.savefig('figures/cs2_scatter.png', dpi=130); plt.show()

# ===== Notebook code cell 19 =====
# Purpose: draw the elbow curve and silhouette scores for K = 2 to 8
ks = range(2, 9); inertia, sil = [], []
for k in ks:
    km_try = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xa)
    inertia.append(km_try.inertia_); sil.append(silhouette_score(Xa, km_try.labels_))
print(pd.DataFrame({'K': list(ks), 'Inertia': np.round(inertia), 'Silhouette': np.round(sil, 3)}).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ks, inertia, 'o-', color='black'); ax[0].set_title('Elbow Curve for K-Means'); ax[0].set_xlabel('Number of clusters (K)'); ax[0].set_ylabel('Inertia (within-cluster squared distance)')
ax[1].plot(ks, sil, 's-', color='black'); ax[1].set_title('Silhouette Score for Different K'); ax[1].set_xlabel('Number of clusters (K)'); ax[1].set_ylabel('Average silhouette score')
plt.tight_layout(); plt.savefig('figures/cs2_elbow.png', dpi=130); plt.show()

# ===== Notebook code cell 21 =====
# Purpose: fit K-Means with K = 4 and describe each cluster using average values in original units
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xa)
air_c['KMeans'] = km.labels_
print('K-Means (K = 4) silhouette score: %.3f' % silhouette_score(Xa, km.labels_))

def profile(labels):
    t = air_c.assign(cluster=labels)
    g = t.groupby('cluster')[feats + ['Award?']].mean().round(2)
    g['members'] = t.groupby('cluster').size()
    g['% of members'] = (g['members'] / len(t) * 100).round(1)
    return g
profile(air_c['KMeans'])

# ===== Notebook code cell 22 =====
# Purpose: reduce the data to two dimensions with PCA so the clusters can be plotted
pca = PCA(n_components=2).fit(Xa); P = pca.transform(Xa)
print('PCA explained variance: PC1 = %.1f%%, PC2 = %.1f%%, together = %.1f%%' % (*(pca.explained_variance_ratio_*100), pca.explained_variance_ratio_.sum()*100))

def plot_pca(labels, title, fname):
    plt.figure(figsize=(7.5, 6))
    sns.scatterplot(x=P[:,0], y=P[:,1], hue=labels, palette='tab10', s=18, alpha=.7, edgecolor=None)
    plt.title(title)
    plt.xlabel('Principal component 1 (%.1f%% of variance)' % (pca.explained_variance_ratio_[0]*100))
    plt.ylabel('Principal component 2 (%.1f%% of variance)' % (pca.explained_variance_ratio_[1]*100))
    plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(fname, dpi=130); plt.show()

plot_pca(km.labels_, 'K-Means Clusters (K = 4) Shown on Two PCA Components', 'figures/cs2_kmeans_clusters.png')

# ===== Notebook code cell 25 =====
# Purpose: try four linkage types and draw the dendrogram for Ward linkage
rows = []
for link in ['ward', 'complete', 'average', 'single']:
    for k in (3, 4):
        lab = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(Xa)
        rows.append({'linkage': link, 'K': k, 'silhouette': round(silhouette_score(Xa, lab), 3), 'cluster sizes': np.bincount(lab).tolist()})
print(pd.DataFrame(rows).to_string(index=False))

Zlink = linkage(Xa, method='ward')
plt.figure(figsize=(11, 5))
dendrogram(Zlink, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=9, color_threshold=0.7*max(Zlink[:,2]))
plt.title('Dendrogram Using Ward Linkage (last 30 merged clusters)')
plt.xlabel('Cluster (number of members in brackets)'); plt.ylabel('Ward distance')
plt.tight_layout(); plt.savefig('figures/cs2_dendrogram.png', dpi=130); plt.show()

# ===== Notebook code cell 27 =====
# Purpose: fit Ward hierarchical clustering with K = 4 and describe each cluster
hc = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(Xa)
air_c['Hierarchical'] = hc.labels_
print('Hierarchical (Ward, K = 4) silhouette score: %.3f' % silhouette_score(Xa, hc.labels_))
display(profile(air_c['Hierarchical']))
plot_pca(hc.labels_, 'Hierarchical Clusters (Ward, K = 4) Shown on Two PCA Components', 'figures/cs2_hierarchical_clusters.png')

# ===== Notebook code cell 30 =====
# Purpose: draw the k-distance plot and try different eps and min_samples values
nn = NearestNeighbors(n_neighbors=8).fit(Xa)
kd = np.sort(nn.kneighbors(Xa)[0][:, -1])
print('k-distance at the 50th, 75th, 90th, 95th and 99th percentile:', np.percentile(kd, [50, 75, 90, 95, 99]).round(2))
plt.figure(figsize=(7.5, 4.5)); plt.plot(kd, color='black'); plt.axhline(1.5, ls='--', color='gray', label='eps = 1.5')
plt.title('k-Distance Plot for Choosing eps in DBSCAN'); plt.xlabel('Members sorted by distance'); plt.ylabel('Distance to the 8th nearest member'); plt.legend()
plt.tight_layout(); plt.savefig('figures/cs2_kdistance.png', dpi=130); plt.show()

rows = []
for eps in (0.3, 0.5, 0.7, 1.0, 1.5):
    for ms in (5, 10):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xa); nc = len(set(lab)) - (1 if -1 in lab else 0); ok = lab != -1
        s = round(silhouette_score(Xa[ok], lab[ok]), 3) if nc > 1 else np.nan
        rows.append({'eps': eps, 'min_samples': ms, 'clusters': nc, 'noise points': int((~ok).sum()), 'silhouette (without noise)': s})
print(pd.DataFrame(rows).to_string(index=False))

# ===== Notebook code cell 32 =====
# Purpose: fit DBSCAN with eps = 1.5 and min_samples = 5 and describe the clusters and noise points
db = DBSCAN(eps=1.5, min_samples=5).fit(Xa)
air_c['DBSCAN'] = db.labels_; db_ok = db.labels_ != -1
print('Clusters found:', len(set(db.labels_)) - 1, '| Noise points:', (~db_ok).sum(), f'({(~db_ok).mean():.1%})')
print('DBSCAN silhouette score (without noise): %.3f' % silhouette_score(Xa[db_ok], db.labels_[db_ok]))
display(profile(air_c['DBSCAN']))
plot_pca(db.labels_, 'DBSCAN Clusters Shown on Two PCA Components (-1 means noise)', 'figures/cs2_dbscan_clusters.png')

# ===== Notebook code cell 35 =====
# Purpose: put the results of the three methods in one table
comparison = pd.DataFrame([
    {'Method': 'K-Means (K = 4)', 'Clusters': 4, 'Noise points': 0, 'Silhouette score': round(silhouette_score(Xa, km.labels_), 3), 'Smallest cluster': int(np.bincount(km.labels_).min())},
    {'Method': 'Hierarchical (Ward, K = 4)', 'Clusters': 4, 'Noise points': 0, 'Silhouette score': round(silhouette_score(Xa, hc.labels_), 3), 'Smallest cluster': int(np.bincount(hc.labels_).min())},
    {'Method': 'DBSCAN (eps = 1.5, min_samples = 5)', 'Clusters': 3, 'Noise points': int((~db_ok).sum()), 'Silhouette score': round(silhouette_score(Xa[db_ok], db.labels_[db_ok]), 3), 'Smallest cluster': int(np.bincount(db.labels_[db_ok]).min())}])
comparison

# ===== Notebook code cell 40 =====
# Purpose: load and combine the red and white wine files, then check size, missing values and duplicates
red   = pd.read_csv(find_data_file('winequality-red.csv'), sep=';')
white = pd.read_csv(find_data_file('winequality-white.csv'), sep=';')
wine = pd.concat([red, white], ignore_index=True)
print('Shape:', wine.shape)
print('Missing values:', wine.isna().sum().sum())
print('Duplicate rows:', wine.duplicated().sum())
wine.describe().T.round(3)

# ===== Notebook code cell 43 =====
# Purpose (Task 1a, 1b): mean, median, mode, standard deviation, skewness and coefficient of variation (CV)
cols = ['alcohol', 'volatile acidity', 'residual sugar']
rows = []
for c in cols:
    s = wine[c]
    rows.append({'attribute': c, 'mean': s.mean(), 'median': s.median(), 'mode': s.mode()[0],
                 'std dev': s.std(), 'skewness': s.skew(), 'CV (%)': s.std()/s.mean()*100})
summary = pd.DataFrame(rows).set_index('attribute').round(3)
summary

# ===== Notebook code cell 45 =====
# Purpose (Task 1c): Pearson correlation between alcohol and quality
r, p = stats.pearsonr(wine['alcohol'], wine['quality'])
print(f'Pearson r (alcohol vs quality) = {r:.3f}, p-value = {p:.2e}')

# ===== Notebook code cell 48 =====
# Purpose (Task 2a): histogram (count plot) of the quality scores
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(x='quality', data=wine, color='steelblue', edgecolor='black')
for bar in ax.patches:
    ax.annotate(int(bar.get_height()), (bar.get_x()+bar.get_width()/2, bar.get_height()), ha='center', va='bottom')
plt.title('Distribution of Wine Quality Scores'); plt.xlabel('Quality score (given by tasters)'); plt.ylabel('Number of wines')
plt.tight_layout(); plt.savefig('figures/cs5_quality_histogram.png', dpi=150); plt.show()
print(wine['quality'].value_counts().sort_index())

# ===== Notebook code cell 50 =====
# Purpose (Task 2b): box plot of alcohol for each quality score, with a count of outliers per score
plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='alcohol', data=wine, color='lightgray', fliersize=3)
plt.title('Alcohol Content for Each Quality Score'); plt.xlabel('Quality score'); plt.ylabel('Alcohol (% by volume)')
plt.tight_layout(); plt.savefig('figures/cs5_alcohol_boxplot.png', dpi=150); plt.show()

def count_outliers(s):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
table = wine.groupby('quality')['alcohol'].agg(['count', 'median'])
table['outliers'] = wine.groupby('quality')['alcohol'].apply(count_outliers)
table

# ===== Notebook code cell 52 =====
# Purpose (Task 2c, 2d): correlation heatmap and ranking of attributes by correlation with quality
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=.5, cbar_kws={'label': 'Pearson correlation'})
plt.title('Correlation Heatmap of Wine Attributes and Quality')
plt.tight_layout(); plt.savefig('figures/cs5_correlation_heatmap.png', dpi=150); plt.show()

qcorr = wine.corr()['quality'].drop('quality')
print(qcorr.reindex(qcorr.abs().sort_values(ascending=False).index).round(3))

# ===== Notebook code cell 55 =====
# Purpose (Task 3a to 3c): 80/20 split, fit multiple linear regression and find MAE, MSE, RMSE and R2 on the test set
X = wine.drop(columns='quality'); y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training rows:', X_train.shape[0], '| Testing rows:', X_test.shape[0])

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
print(f'MAE  = {mae:.4f}\nMSE  = {mse:.4f}\nRMSE = {rmse:.4f}\nR2   = {r2:.4f}')

baseline = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
print(f'MAE if we always guess the average training score = {baseline:.4f}')
print(f'Predictions within 0.5 of the true score: {np.mean(abs(y_pred-y_test)<=0.5):.1%}')
print(f'Predictions within 1.0 of the true score: {np.mean(abs(y_pred-y_test)<=1):.1%}')

# ===== Notebook code cell 57 =====
# Purpose (Task 3d): standardised coefficients to compare predictors, and VIF to check multicollinearity
scaler = StandardScaler().fit(X_train)
model_std = LinearRegression().fit(scaler.transform(X_train), y_train)
std_coef = pd.Series(model_std.coef_, index=X.columns)
std_coef = std_coef.reindex(std_coef.abs().sort_values(ascending=False).index)
print('Standardised coefficients:\n', std_coef.round(3))

Xc = np.column_stack([np.ones(len(X_train)), X_train.values])
vif = pd.Series([variance_inflation_factor(Xc, i+1) for i in range(X.shape[1])], index=X.columns).sort_values(ascending=False)
print('\nVariance Inflation Factors (VIF):\n', vif.round(2))

plt.figure(figsize=(7, 4.5))
plt.barh(std_coef.index[::-1], std_coef.values[::-1], color='gray', edgecolor='black')
plt.axvline(0, color='black', lw=.8)
plt.title('Standardised Regression Coefficients'); plt.xlabel('Change in quality for a one standard deviation rise in the attribute'); plt.ylabel('Attribute')
plt.tight_layout(); plt.savefig('figures/cs5_standardised_coefficients.png', dpi=150); plt.show()

# ===== Notebook code cell 59 =====
# Purpose (Task 3e): scatter plot of predicted versus actual quality on the test set
plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, y_pred, alpha=.35, color='gray', edgecolor='black', s=18)
plt.plot([3, 9], [3, 9], 'k--', label='Perfect prediction')
plt.title('Predicted vs Actual Wine Quality (Test Set)'); plt.xlabel('Actual quality score'); plt.ylabel('Predicted quality score')
plt.legend(); plt.tight_layout(); plt.savefig('figures/cs5_predicted_vs_actual.png', dpi=150); plt.show()
print('Predictions range from', y_pred.min().round(2), 'to', y_pred.max().round(2))

# ===== Notebook code cell 62 =====
# Purpose (extra check): repeat the model after removing duplicate rows, and after adding a red/white wine type column
red['wine_type'] = 0; white['wine_type'] = 1
wine_t = pd.concat([red, white], ignore_index=True)
feats11 = list(X.columns)

# how many test wines in the main model have an identical twin in the training set?
train_keys = set(map(tuple, X_train.round(6).values))
share_twin = np.mean([tuple(row) in train_keys for row in X_test.round(6).values])
print(f'Test wines that also appear in the training set: {share_twin:.1%}')

def evaluate(data, cols, name):
    Xs, ys = data[cols], data['quality']
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42)
    pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
    mse_ = mean_squared_error(yte, pred)
    return {'Model': name, 'Rows used': len(data), 'Test rows': len(yte), 'MAE': round(mean_absolute_error(yte, pred), 3),
            'MSE': round(mse_, 3), 'RMSE': round(np.sqrt(mse_), 3), 'R2': round(r2_score(yte, pred), 3)}

unique = wine_t.drop_duplicates(subset=feats11 + ['quality']).reset_index(drop=True)
check = pd.DataFrame([
    evaluate(wine_t, feats11, 'Main model (all rows, 11 attributes)'),
    evaluate(unique, feats11, 'Duplicates removed, 11 attributes'),
    evaluate(wine_t, feats11 + ['wine_type'], 'All rows, 11 attributes + wine type'),
    evaluate(unique, feats11 + ['wine_type'], 'Duplicates removed + wine type')])
check