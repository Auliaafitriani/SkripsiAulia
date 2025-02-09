import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import random
from sklearn.manifold import TSNE

# [Previous class and function definitions remain the same - include all the classes and functions from the original code]

def main():
    # Sidebar for navigation
    with st.sidebar:
        selected = option_menu('Government Aid Priorities Dashboard',
                           ['About', 'Upload Data', 'Preprocessing', 
                            'PSO and K-Medoids Results'],
                           menu_icon='cast',
                           icons=['house', 'cloud-upload', 'gear', 'graph-up'],
                           default_index=0)
        
                # Add copyright at the bottom
        st.markdown("---")
        st.markdown("© 2025 Copyright by Aulia Nur Fitriani")

    if selected == 'About':
        st.title('Government Aid Priorities Dashboard')
        st.write("""
        Optimasi K-Medoids dengan Particle Swarm Optimization bertujuan untuk menentukan prioritas bantuan pemerintah di Desa Kalipuro. 
        Proyek ini menggunakan metode K-Medoids untuk mengelompokkan data dan Particle Swarm Optimization untuk meningkatkan efisiensi pengelompokan. 
        Dashboard ini dirancang untuk membantu pemangku kepentingan dalam memahami data dan hasil analisis yang dilakukan. 
        Dengan menggunakan visualisasi interaktif, pengguna dapat mengeksplorasi data dan hasil pengelompokan dengan lebih baik.
        """)

    elif selected == 'Upload Data':
        st.title('Upload Data')
        uploaded_file = st.file_uploader("Pilih file Excel", type=['xlsx'])
        
        if uploaded_file is not None:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            st.write("### Data Asli")
            st.dataframe(df)
            # Save to session state
            st.session_state['original_data'] = df
            st.success('Data berhasil diunggah!')

    elif selected == 'Preprocessing':
        st.title('Data Preprocessing')
        
        def weighted_normalize(df):
            # Simpan kolom ID dan PEKERJAAN sebagai DataFrame terpisah
            id_column = df[['ID', 'PEKERJAAN']]

            # Pilih hanya kolom numerik untuk normalisasi
            df_numeric = df.drop(columns=['ID', 'PEKERJAAN'])

            # Definisi bobot
            weights = {
                'JUMLAH ASET MOBIL': 4,
                'JUMLAH ASET MOTOR': 1,
                'JUMLAH ASET RUMAH/TANAH/SAWAH': 5,
                'PENDAPATAN': 6
            }

            # Inisialisasi MinMaxScaler
            scaler = MinMaxScaler()

            # Normalisasi kolom numerik
            normalized_data = scaler.fit_transform(df_numeric)
            df_normalized = pd.DataFrame(normalized_data, columns=df_numeric.columns)

            # Terapkan bobot
            for col in df_normalized.columns:
                if col in weights:
                    df_normalized[col] *= weights[col]

            # Gabungkan kembali kolom ID dan PEKERJAAN
            df_normalized = pd.concat([id_column, df_normalized], axis=1)

            return df_normalized
        
        if 'original_data' not in st.session_state or st.session_state['original_data'] is None:
            st.warning('Silakan upload data terlebih dahulu pada halaman Upload Data')
            return
            
        if st.button('Lakukan Preprocessing'):
            try:
                df = st.session_state['original_data']

                # Statistik Deskriptif
                st.write("### Statistik Deskriptif")
                # Pilih kolom numerik untuk statistik deskriptif
                numeric_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
                st.dataframe(df[numeric_columns].describe())
                
                # Check Missing Values
                st.write("### Pengecekan Missing Values")
                missing_values = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Kolom': missing_values.index,
                    'Jumlah Missing Values': missing_values.values
                })
                st.dataframe(missing_df)
                
                # Check Outliers
                st.write("### Pengecekan Outliers")
                # Pilih kolom numerik untuk analisis outlier
                selected_columns = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
                
                # Hitung Q1, Q3, dan IQR
                Q1 = df[selected_columns].quantile(0.25)
                Q3 = df[selected_columns].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Hitung jumlah outlier
                outliers = ((df[selected_columns] < lower_bound) | (df[selected_columns] > upper_bound)).sum()
                outlier_df = pd.DataFrame({
                    'Kolom': outliers.index,
                    'Jumlah Outlier': outliers.values
                })
                st.dataframe(outlier_df)
                
                # Lakukan normalisasi
                st.write("### Data Setelah Normalisasi dan Pembobotan")
                df_normalized = weighted_normalize(df)
                st.session_state['df_normalized'] = df_normalized
                st.dataframe(df_normalized)
                
                st.success('Preprocessing selesai!')
            except Exception as e:
                st.error(f'Error saat preprocessing: {str(e)}')

    elif selected == 'PSO and K-Medoids Results':
            st.title('PSO and K-Medoids Analysis')
    
            # Tampilkan parameter PSO
            st.write("### Parameter PSO yang Digunakan")
            st.write("""
            - w (inertia weight) = 0.7
            - c1 (cognitive coefficient) = 1.5
            - c2 (social coefficient) = 1.5
            - Jumlah Partikel = 30
            - Maksimum Iterasi = 100
            """)

            if 'df_normalized' not in st.session_state:
                st.warning('Silakan lakukan preprocessing terlebih dahulu')
                return
            
            # Kolom untuk interaksi
            col1, col2 = st.columns(2)
            
            with col1:
                # Slider untuk memilih K
                n_clusters = st.slider('Jumlah Cluster (K)', min_value=2, max_value=5, value=2)
            
            with col2:
                # Tombol untuk analisis
                analyze_button = st.button(f'Analisis dengan K={n_clusters}')
            
            # Tambahkan kolom untuk menampilkan hasil evaluasi
            if analyze_button:
                with st.spinner('Sedang melakukan clustering...'):
                    df_clustered, cluster_info = perform_clustering(st.session_state['df_normalized'], n_clusters)
                    
                    # Simpan hasil untuk perbandingan
                    if 'clustering_results' not in st.session_state:
                        st.session_state['clustering_results'] = {}
                    
                    st.session_state['clustering_results'][n_clusters] = {
                        'silhouette': cluster_info['silhouette_score'],
                        'distribution': cluster_info['cluster_sizes']
                    }
                    
                    # Tampilkan hasil clustering
                    st.write("### Hasil Clustering")
                    st.dataframe(df_clustered)
                    
                    # Tampilkan evaluasi metrics
                    st.write("### Evaluasi Clustering")
                    st.write(f"Silhouette Score untuk K={n_clusters}: {cluster_info['silhouette_score']:.4f}")
                    
                    # Tampilkan distribusi cluster
                    st.write("### Distribusi Cluster:")
                    for i, count in enumerate(cluster_info['cluster_sizes']):
                        st.write(f"Cluster {i}: {count} titik data")
                    
                    # Visualisasi distribusi
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(cluster_info['cluster_sizes'])), cluster_info['cluster_sizes'])
                    plt.title(f'Distribusi Anggota Cluster (K={n_clusters})')
                    plt.xlabel('Cluster')
                    plt.ylabel('Jumlah Anggota')
                    st.pyplot(plt)
                    
                    # Visualisasi t-SNE
                    st.write("### Visualisasi Cluster")
                    plt_tsne = visualize_kmedoids_clusters(df_clustered, cluster_info)
                    st.pyplot(plt_tsne)
                    
                    # Tampilkan perbandingan silhouette score
                    if len(st.session_state['clustering_results']) > 0:
                        st.write("### Perbandingan Silhouette Score")
                        comparison_data = {
                            'K': list(st.session_state['clustering_results'].keys()),
                            'Silhouette Score': [info['silhouette'] for info in st.session_state['clustering_results'].values()]
                        }
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)
                        
                        # Visualisasi perbandingan
                        plt.figure(figsize=(10, 6))
                        plt.plot(comparison_df['K'], comparison_df['Silhouette Score'], marker='o')
                        plt.title('Perbandingan Silhouette Score untuk Berbagai Nilai K')
                        plt.xlabel('Jumlah Cluster (K)')
                        plt.ylabel('Silhouette Score')
                        plt.grid(True)
                        st.pyplot(plt)
                    
                    # Best medoids information
                    st.write("### Medoid Terbaik")
                    st.dataframe(cluster_info['medoid_rows'])
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = df_clustered.to_csv(index=False)
                        st.download_button(
                            label=f"Download Hasil K={n_clusters} (CSV)",
                            data=csv,
                            file_name=f'hasil_clustering_k{n_clusters}.csv',
                            mime='text/csv'
                        )
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def perform_clustering(df, n_clusters):
    # Pisahkan kolom yang tidak akan di-cluster
    non_cluster_columns = ['ID', 'PEKERJAAN']
    cluster_columns = [col for col in df.columns if col not in non_cluster_columns]
    
    # Persiapkan data untuk clustering
    X = df[cluster_columns].values
    
    # Standarisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Inisialisasi medoids secara acak
    medoid_indices = np.random.choice(len(X_scaled), n_clusters, replace=False)
    
    # Fungsi untuk menghitung jarak
    def calculate_distances(X, medoids):
        distances = np.zeros((len(X), len(medoids)))
        for i, medoid in enumerate(medoids):
            distances[:, i] = np.linalg.norm(X - X[medoid], axis=1)
        return distances
    
    # Iterasi untuk memperbaiki medoids
    max_iterations = 100
    for _ in range(max_iterations):
        # Assign cluster
        distances = calculate_distances(X_scaled, medoid_indices)
        clusters = np.argmin(distances, axis=1)
        
        # Update medoids
        new_medoids = medoid_indices.copy()
        for i in range(n_clusters):
            cluster_points = X_scaled[clusters == i]
            if len(cluster_points) > 0:
                # Cari titik terdekat dengan rata-rata cluster sebagai medoid baru
                cluster_center = np.mean(cluster_points, axis=0)
                new_medoid_idx = np.argmin(np.linalg.norm(cluster_points - cluster_center, axis=1))
                new_medoids[i] = np.where((X_scaled == cluster_points[new_medoid_idx]).all(axis=1))[0][0]
        
        # Cek apakah medoids tidak berubah
        if np.array_equal(new_medoids, medoid_indices):
            break
        
        medoid_indices = new_medoids
    
    # Hitung silhouette score
    try:
        silhouette_avg = silhouette_score(X_scaled, clusters)
    except:
        silhouette_avg = 0
    
    # Tambahkan kolom cluster ke dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # Informasi cluster
    cluster_info = {
        'silhouette_score': silhouette_avg,
        'cluster_sizes': [np.sum(clusters == i) for i in range(n_clusters)],
        'medoid_rows': df.iloc[medoid_indices],
        'medoid_indices': medoid_indices
    }
    
    return df_clustered, cluster_info

def visualize_kmedoids_clusters(df_clustered, cluster_info):
    # Ekstrak data untuk visualisasi
    cluster_columns = [col for col in df_clustered.columns if col not in ['ID', 'PEKERJAAN', 'Cluster']]
    X = df_clustered[cluster_columns].values
    
    # Reduksi dimensi menggunakan t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Warna untuk cluster
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    
    # Plot titik-titik cluster
    for i in range(len(np.unique(df_clustered['Cluster']))):
        mask = df_clustered['Cluster'] == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    c=colors[i], 
                    label=f'Cluster {i}', 
                    alpha=0.7)
    
    # Plot medoids
    medoid_indices = cluster_info['medoid_indices']
    plt.scatter(X_tsne[medoid_indices, 0], X_tsne[medoid_indices, 1], 
                c='black', 
                marker='*', 
                s=300, 
                label='Medoids')
    
    # Tambahkan garis dari setiap titik ke medoidnya
    for i, medoid_idx in enumerate(medoid_indices):
        cluster_mask = df_clustered['Cluster'] == i
        plt.plot(
            np.column_stack([X_tsne[cluster_mask, 0], np.repeat(X_tsne[medoid_idx, 0], sum(cluster_mask))]),
            np.column_stack([X_tsne[cluster_mask, 1], np.repeat(X_tsne[medoid_idx, 1], sum(cluster_mask))]),
            c='gray', 
            alpha=0.3, 
            linewidth=0.5
        )
    
    plt.title('K-Medoids Clustering Visualization', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return plt
    
main()
