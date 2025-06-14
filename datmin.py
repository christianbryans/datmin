"""
Analisis Data Lagu Spotify 2024
==============================

Permasalahan:
- Perlu memahami pola dan karakteristik lagu-lagu yang populer di Spotify
- Memprediksi apakah sebuah lagu berpotensi masuk Top 100 berdasarkan fitur-fiturnya

Tujuan:
1. Mengelompokkan lagu berdasarkan karakteristiknya (clustering)
2. Memprediksi potensi lagu masuk Top 100 (klasifikasi)
3. Mengidentifikasi faktor-faktor yang mempengaruhi popularitas lagu

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
import joblib

# Set page config
st.set_page_config(
    page_title="Spotify Song Analysis",
    page_icon="🎵",
    layout="wide"
)

# Cache data loading
@st.cache_data
def load_data():
    file_path = "Most Streamed Spotify Songs 2024.csv"
    return pd.read_csv(file_path, encoding='cp1252')

# Cache data preprocessing
@st.cache_data
def preprocess_data(df):
    numeric_columns = [
        'Track Score', 'Spotify Streams', 'Spotify Playlist Count', 
        'Spotify Playlist Reach', 'Spotify Popularity', 'YouTube Views', 
        'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 'TikTok Views'
    ]
    
    # Data Cleaning
    for col in numeric_columns:
        df[col] = df[col].astype(str).str.replace(',', '').replace('', '0').astype(float)
    df = df.fillna(0)
    
    return df, numeric_columns

# Cache clustering results
@st.cache_data
def perform_clustering(df, features_for_clustering):
    scaler = StandardScaler()
    X_cluster = df[features_for_clustering]
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    return df, kmeans, scaler

# Cache classification results
@st.cache_data
def train_classification_model(df, features_for_classification):
    df['All Time Rank'] = pd.to_numeric(df['All Time Rank'], errors='coerce').fillna(0).astype(int)
    df['Top100'] = (df['All Time Rank'] <= 100).astype(int)
    
    X_class = df[features_for_classification]
    y_class = df['Top100']
    
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    return log_reg, scaler, X_test_scaled, y_test

def create_dashboard():
    st.title("Analisis Data Lagu Spotify 2024")
    
    # Load and preprocess data
    spotify_df, numeric_columns = preprocess_data(load_data())
    
    # Sidebar for navigation
    st.sidebar.title("Menu Navigasi")
    page = st.sidebar.radio("Pilih Halaman:", ["Data Overview", "Clustering Analysis", "Classification Analysis"])
    
    if page == "Data Overview":
        st.header("Data Overview")
        
        # Display basic statistics
        st.subheader("Statistik Dasar")
        st.dataframe(spotify_df.describe())
        
        # Display correlation matrix
        st.subheader("Korelasi Antar Fitur")
        correlation_matrix = spotify_df[numeric_columns].corr()
        fig = px.imshow(correlation_matrix,
                       labels=dict(color="Correlation"),
                       x=correlation_matrix.columns,
                       y=correlation_matrix.columns)
        st.plotly_chart(fig)
        
    elif page == "Clustering Analysis":
        st.header("Analisis Clustering")
        
        # Penjelasan Clustering
        st.markdown("""
        ### Apa itu Clustering?
        Clustering adalah teknik untuk mengelompokkan lagu-lagu yang memiliki karakteristik serupa. 
        Dalam analisis ini, kita menggunakan K-Means Clustering untuk membagi lagu menjadi 2 kelompok:
        
        #### Fitur yang Digunakan:
        - **Track Score**: Skor keseluruhan lagu
        - **Spotify Streams**: Jumlah streaming di Spotify
        - **Spotify Playlist Count**: Jumlah playlist yang memuat lagu
        - **Spotify Playlist Reach**: Jangkauan playlist yang memuat lagu
        
        #### Hasil Pengelompokan:
        - **Cluster 0**: Lagu dengan performa sedang
        - **Cluster 1**: Lagu dengan performa tinggi
        
        Visualisasi di bawah ini menunjukkan bagaimana lagu-lagu dikelompokkan berdasarkan karakteristiknya.
        """)
        
        features_for_clustering = ['Track Score', 'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach']
        spotify_df, kmeans, scaler = perform_clustering(spotify_df, features_for_clustering)
        
        # Display cluster visualization
        st.subheader("Visualisasi Cluster")
        pca = PCA(n_components=2)
        X_cluster = spotify_df[features_for_clustering]
        X_cluster_scaled = scaler.transform(X_cluster)
        X_pca = pca.fit_transform(X_cluster_scaled)
        
        fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1],
                        color=spotify_df['Cluster'],
                        title='Visualisasi Pengelompokan Lagu')
        st.plotly_chart(fig)
        
        # Menampilkan statistik cluster
        st.subheader("Statistik per Cluster")
        cluster_stats = spotify_df.groupby('Cluster')[features_for_clustering].mean()
        st.dataframe(cluster_stats)
        
        # Interpretasi cluster
        st.markdown("""
        ### Interpretasi Hasil Clustering
        
        #### Cluster 0 (Performa Sedang):
        - Memiliki nilai fitur yang lebih rendah
        - Biasanya lagu-lagu dengan popularitas menengah
        - Memiliki jumlah streaming dan playlist yang lebih sedikit
        
        #### Cluster 1 (Performa Tinggi):
        - Memiliki nilai fitur yang lebih tinggi
        - Biasanya lagu-lagu yang sangat populer
        - Memiliki jumlah streaming dan playlist yang lebih banyak
        
        ### Manfaat Analisis Clustering:
        1. **Pemahaman Pasar**: Membantu memahami segmentasi lagu di Spotify
        2. **Strategi Marketing**: Membantu dalam pengambilan keputusan marketing
        3. **Analisis Performa**: Memudahkan analisis performa lagu
        """)
        
    else:  # Classification Analysis
        st.header("Analisis Klasifikasi")
        
        # Penjelasan Klasifikasi
        st.markdown("""
        ### Apa itu Klasifikasi?
        Klasifikasi adalah teknik untuk memprediksi apakah sebuah lagu berpotensi masuk Top 100 
        berdasarkan karakteristiknya. Dalam analisis ini, kita menggunakan Logistic Regression.
        
        #### Fitur yang Digunakan:
        - **Track Score**: Skor keseluruhan lagu
        - **Spotify Streams**: Jumlah streaming di Spotify
        - **Spotify Playlist Count**: Jumlah playlist yang memuat lagu
        - **Spotify Playlist Reach**: Jangkauan playlist yang memuat lagu
        
        #### Target:
        - **Top 100**: Lagu yang masuk dalam 100 lagu teratas
        - **Non-Top 100**: Lagu yang tidak masuk dalam 100 lagu teratas
        """)
        
        features_for_classification = ['Track Score', 'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach']
        model, scaler, X_test, y_test = train_classification_model(spotify_df, features_for_classification)
        
        # Display model performance
        st.subheader("Model Performance")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Menampilkan akurasi
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Akurasi Model", f"{accuracy:.2%}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
        fig.update_layout(title='Kurva ROC',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        st.plotly_chart(fig)
        
        # Feature importance
        st.subheader("Pentingnya Fitur")
        importance_df = pd.DataFrame({
            'Feature': features_for_classification,
            'Importance': model.coef_[0]
        })
        fig = px.bar(importance_df, x='Feature', y='Importance',
                    title='Pentingnya Fitur dalam Prediksi')
        st.plotly_chart(fig)
        
        # Interpretasi model
        st.markdown("""
        ### Interpretasi Hasil Klasifikasi
        
        #### Akurasi Model:
        - Menunjukkan seberapa baik model memprediksi lagu Top 100
        - Semakin tinggi akurasi, semakin baik model dalam memprediksi
        
        #### Pentingnya Fitur:
        - **Nilai Positif**: Fitur yang meningkatkan kemungkinan lagu masuk Top 100
        - **Nilai Negatif**: Fitur yang menurunkan kemungkinan lagu masuk Top 100
        
        #### Kurva ROC:
        - Menunjukkan kemampuan model dalam membedakan lagu Top 100 dan Non-Top 100
        - Semakin tinggi AUC (Area Under Curve), semakin baik model dalam membedakan
        
        ### Manfaat Analisis Klasifikasi:
        1. **Prediksi Popularitas**: Membantu memprediksi potensi lagu masuk Top 100
        2. **Pengambilan Keputusan**: Membantu dalam pengambilan keputusan bisnis
        3. **Optimasi Strategi**: Membantu mengoptimalkan strategi marketing
        """)

if __name__ == "__main__":
    create_dashboard()

