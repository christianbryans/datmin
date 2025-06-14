"""
Analisis Data Lagu Spotify 2024
==============================

[PLO01 - CLO05] Business Understanding
-------------------------------------
Permasalahan:
- Perlu memahami pola dan karakteristik lagu-lagu yang populer di Spotify
- Memprediksi apakah sebuah lagu berpotensi masuk Top 100 berdasarkan fitur-fiturnya

Tujuan:
1. Mengelompokkan lagu berdasarkan karakteristiknya (clustering)
2. Memprediksi potensi lagu masuk Top 100 (klasifikasi)
3. Mengidentifikasi faktor-faktor yang mempengaruhi popularitas lagu

[PLO02 - CLO02] Data Preparation
-------------------------------
Tahapan yang dilakukan:
1. Data Cleaning: menghapus koma, mengisi nilai kosong
2. Data Transformation: konversi tipe data
3. Feature Scaling: standarisasi data
4. Feature Selection: pemilihan fitur yang relevan
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc, precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# =============================================
# 1. LOADING DAN PREPROCESSING DATA
# =============================================

# Membuat direktori untuk menyimpan chart
output_dir = "output_charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Membaca dataset
file_path = "Most Streamed Spotify Songs 2024.csv"
spotify_df = pd.read_csv(file_path, encoding='cp1252')

# Menampilkan informasi dataset
print("\nInformasi Dataset:")
print(spotify_df.info())
print("\nStatistik Deskriptif:")
print(spotify_df.describe())

# Mendefinisikan kolom numerik
numeric_columns = [
    'Track Score', 'Spotify Streams', 'Spotify Playlist Count', 
    'Spotify Playlist Reach', 'Spotify Popularity', 'YouTube Views', 
    'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 'TikTok Views'
]

# Data Cleaning
for col in numeric_columns:
    spotify_df[col] = spotify_df[col].astype(str).str.replace(',', '').replace('', '0').astype(float)
spotify_df = spotify_df.fillna(0)

# =============================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================

# Distribusi fitur numerik
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(data=spotify_df, x=col, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
plt.close()

# Korelasi antar fitur
plt.figure(figsize=(12, 8))
correlation_matrix = spotify_df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# Box plot untuk outlier detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(data=spotify_df, y=col)
    plt.title(f'Box Plot {col}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplots.png'))
plt.close()

# =============================================
# 3. UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# =============================================

# Pemilihan fitur untuk clustering
features_for_clustering = ['Track Score', 'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach']
X_cluster = spotify_df[features_for_clustering]

# Standarisasi data
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Mencari jumlah cluster optimal
inertias = []
silhouette_scores = []
K = range(2, 11)

print("\nAnalisis Jumlah Cluster Optimal:")
print("================================")
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))
    print(f"K={k} --> Silhouette Score: {silhouette_scores[-1]:.4f}")

# Visualisasi metode Elbow dan Silhouette
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'elbow_silhouette.png'))
plt.close()

# Clustering dengan K=2
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
spotify_df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Analisis detail cluster
cluster_analysis = {
    'Cluster_Size': spotify_df['Cluster'].value_counts(),
    'Cluster_Means': spotify_df.groupby('Cluster')[features_for_clustering].mean(),
    'Cluster_Std': spotify_df.groupby('Cluster')[features_for_clustering].std(),
    'Cluster_Min': spotify_df.groupby('Cluster')[features_for_clustering].min(),
    'Cluster_Max': spotify_df.groupby('Cluster')[features_for_clustering].max()
}

# Print cluster means untuk verifikasi
print("\nRata-rata fitur per cluster:")
print(cluster_analysis['Cluster_Means'])

# Menentukan label cluster berdasarkan nilai rata-rata
cluster_means = cluster_analysis['Cluster_Means']
high_performance_cluster = cluster_means.mean(axis=1).idxmax()

# Menamai cluster berdasarkan analisis
cluster_names = {
    high_performance_cluster: "High Performance",
    1 - high_performance_cluster: "Moderate Performance"
}
spotify_df['Cluster_Name'] = spotify_df['Cluster'].map(cluster_names)

# Visualisasi cluster
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=spotify_df['Cluster'], cmap='Set1', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Cluster Visualization')
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_visualization.png'))
plt.close()

# Visualisasi karakteristik cluster
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_for_clustering, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Cluster', y=feature, data=spotify_df)
    plt.title(f'{feature} by Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_characteristics.png'))
plt.close()

# Menyimpan hasil analisis cluster
cluster_results = {
    'cluster_analysis': cluster_analysis,
    'cluster_names': cluster_names,
    'cluster_centers': kmeans.cluster_centers_,
    'silhouette_score': silhouette_score(X_cluster_scaled, kmeans.labels_)
}

# =============================================
# 4. SUPERVISED LEARNING - LOGISTIC REGRESSION
# =============================================

# Mempersiapkan target variable
spotify_df['All Time Rank'] = pd.to_numeric(spotify_df['All Time Rank'], errors='coerce').fillna(0).astype(int)
spotify_df['Top100'] = (spotify_df['All Time Rank'] <= 100).astype(int)

# Pemilihan fitur untuk klasifikasi
features_for_classification = ['Track Score', 'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach']
X_class = spotify_df[features_for_classification]
y_class = spotify_df['Top100']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Standarisasi fitur
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training model Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Evaluasi model
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Menyimpan hasil evaluasi
evaluation_results = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Classification Report': classification_report(y_test, y_pred),
    'Feature Importance': dict(zip(features_for_classification, log_reg.coef_[0]))
}

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# =============================================
# 5. DASHBOARD DENGAN STREAMLIT
# =============================================

def create_dashboard():
    st.title('Spotify Data Analysis 2024')
    
    # Sidebar
    st.sidebar.header('Navigation')
    page = st.sidebar.selectbox('Select Page', ['Overview', 'Clustering', 'Classification'])
    
    if page == 'Overview':
        st.header('Data Overview')
        
        # Statistik dasar
        st.subheader('Statistik Deskriptif')
        st.write(spotify_df.describe())
        
        # Distribusi fitur
        st.subheader('Feature Distribution')
        feature = st.selectbox('Select Feature', numeric_columns)
        fig = px.histogram(spotify_df, x=feature, title=f'Distribution of {feature}')
        st.plotly_chart(fig)
        
        # Korelasi antar fitur
        st.subheader('Feature Correlations')
        fig = px.imshow(correlation_matrix, 
                       labels=dict(color="Correlation"),
                       title='Feature Correlations')
        st.plotly_chart(fig)
        
    elif page == 'Clustering':
        st.header('Cluster Analysis')
        
        # Penjelasan metode
        st.write("""
        ### Metode Clustering
        K-Means clustering digunakan untuk mengelompokkan lagu-lagu berdasarkan karakteristiknya.
        Fitur yang digunakan:
        - Track Score
        - Spotify Streams
        - Spotify Playlist Count
        - Spotify Playlist Reach
        """)
        
        # Visualisasi cluster
        st.subheader('Cluster Visualization')
        fig = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=spotify_df['Cluster_Name'],
            title='Cluster Distribution'
        )
        st.plotly_chart(fig)
        
        # Karakteristik cluster
        st.subheader('Cluster Statistics')
        
        # Menampilkan statistik cluster
        st.write("Statistik per Cluster:")
        st.write(cluster_analysis['Cluster_Means'])
        
        # Menampilkan interpretasi berdasarkan data aktual
        high_cluster = cluster_analysis['Cluster_Means'].loc[high_performance_cluster]
        moderate_cluster = cluster_analysis['Cluster_Means'].loc[1 - high_performance_cluster]
        
        st.write(f"""
        ### Cluster {high_performance_cluster}: High Performance
        - Track Score: {high_cluster['Track Score']:.2f}
        - Spotify Streams: {high_cluster['Spotify Streams']:.2f}
        - Spotify Playlist Count: {high_cluster['Spotify Playlist Count']:.2f}
        - Spotify Playlist Reach: {high_cluster['Spotify Playlist Reach']:.2f}
        
        ### Cluster {1-high_performance_cluster}: Moderate Performance
        - Track Score: {moderate_cluster['Track Score']:.2f}
        - Spotify Streams: {moderate_cluster['Spotify Streams']:.2f}
        - Spotify Playlist Count: {moderate_cluster['Spotify Playlist Count']:.2f}
        - Spotify Playlist Reach: {moderate_cluster['Spotify Playlist Reach']:.2f}
        """)
        
        # Visualisasi karakteristik
        st.subheader('Feature Distribution by Cluster')
        for feature in features_for_clustering:
            fig = px.box(spotify_df, x='Cluster_Name', y=feature,
                        title=f'{feature} by Cluster')
            st.plotly_chart(fig)
        
        # Interpretasi cluster
        st.subheader('Interpretasi Cluster')
        st.write("""
        ### Mengapa Terbentuk 2 Cluster?
        1. **Pemisahan Natural**: Data lagu Spotify secara natural terbagi menjadi dua kelompok berdasarkan performa
        2. **Karakteristik Berbeda**: 
           - Cluster High Performance: Lagu-lagu dengan nilai fitur yang lebih tinggi
           - Cluster Moderate Performance: Lagu-lagu dengan nilai fitur yang lebih rendah
        3. **Silhouette Score**: Nilai silhouette score yang tinggi menunjukkan pemisahan cluster yang baik
        
        ### Manfaat Pemisahan Cluster
        1. **Pemahaman Pasar**: Membantu memahami segmentasi lagu di Spotify
        2. **Strategi Marketing**: Membantu dalam pengambilan keputusan marketing
        3. **Analisis Performa**: Memudahkan analisis performa lagu
        """)
        
    else:  # Klasifikasi
        st.header('Classification Results')
        
        # Penjelasan model
        st.write("""
        Model ini memprediksi apakah sebuah lagu berpotensi masuk Top 100 berdasarkan fitur-fiturnya.
        """)
        
        # Hasil evaluasi
        st.subheader('Model Performance')
        st.write(f"Accuracy: {evaluation_results['Accuracy']:.2%}")
        
        # Classification Report
        st.subheader('Classification Report')
        st.text(evaluation_results['Classification Report'])
        
        # Feature Importance
        st.subheader('Feature Importance')
        importance_df = pd.DataFrame({
            'Feature': features_for_classification,
            'Coefficient': evaluation_results['Feature Importance'].values()
        })
        fig = px.bar(importance_df, x='Feature', y='Coefficient',
                    title='Feature Importance')
        st.plotly_chart(fig)
        
        # ROC Curve
        st.subheader('ROC Curve')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC curve'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig)
        
        # Interpretasi
        st.subheader('Interpretasi Model')
        st.write("""
        1. **Akurasi**: Menunjukkan seberapa baik model memprediksi lagu Top 100
        2. **Feature Importance**: 
           - Koefisien positif: fitur yang meningkatkan kemungkinan lagu masuk Top 100
           - Koefisien negatif: fitur yang menurunkan kemungkinan lagu masuk Top 100
        3. **ROC Curve**: Menunjukkan trade-off antara True Positive Rate dan False Positive Rate
        """)

if __name__ == '__main__':
    create_dashboard()

