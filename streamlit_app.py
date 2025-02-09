import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import random
from sklearn.manifold import TSNE

# [Previous class and function definitions remain the same - include all the classes and functions from the original code]
class PSOKMedoids:
    def __init__(self, data, n_clusters, max_iter=100, n_particles=30, w=0.7, c1=1.5, c2=1.5):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def calculate_cost(self, medoids):
        total_cost = 0
        for point in self.data:
            distances = [self.euclidean_distance(point, self.data[medoid]) for medoid in medoids]
            total_cost += min(distances)
        return total_cost

    def evaluate_clustering(self, medoids):
        labels = np.zeros(len(self.data), dtype=int)
        for i, point in enumerate(self.data):
            distances = [self.euclidean_distance(point, self.data[medoid]) for medoid in medoids]
            labels[i] = np.argmin(distances)

        silhouette_avg = silhouette_score(self.data, labels)
        cluster_counts = np.bincount(labels)

        return silhouette_avg, labels, cluster_counts

    def optimize(self):
        # Initialize particles
        particles = [random.sample(range(len(self.data)), self.n_clusters) for _ in range(self.n_particles)]
        personal_best = particles.copy()
        personal_best_cost = [self.calculate_cost(p) for p in particles]
        global_best = particles[np.argmin(personal_best_cost)]
        global_best_cost = min(personal_best_cost)
        velocities = [np.zeros(self.n_clusters) for _ in range(self.n_particles)]

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                current_cost = self.calculate_cost(particles[i])
                
                if current_cost < personal_best_cost[i]:
                    personal_best[i] = particles[i]
                    personal_best_cost[i] = current_cost
                if current_cost < global_best_cost:
                    global_best = particles[i]
                    global_best_cost = current_cost

                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * np.array(velocities[i]) +
                    self.c1 * r1 * (np.array(personal_best[i]) - np.array(particles[i])) +
                    self.c2 * r2 * (np.array(global_best) - np.array(particles[i]))
                )
                particles[i] = [
                    max(0, min(len(self.data)-1, int(p + v)))
                    for p, v in zip(particles[i], velocities[i])
                ]
                particles[i] = list(set(particles[i]))
                while len(particles[i]) < self.n_clusters:
                    particles[i].append(random.randint(0, len(self.data)-1))

        return global_best, *self.evaluate_clustering(global_best)
        
def perform_clustering(df_normalized, n_clusters=5):
    # Pisahkan kolom yang tidak akan di-cluster
    data = df_normalized.drop(columns=['ID', 'PEKERJAAN']).values
    
    # Gunakan kelas PSOKMedoids untuk clustering
    pso = PSOKMedoids(data, n_clusters=n_clusters)
    
    # Optimize dan dapatkan hasil clustering
    optimal_medoids, silhouette, labels, distribution = pso.optimize()

    # Tambahkan kolom cluster ke dataframe
    df_clustered = df_normalized.copy()
    df_clustered['Cluster'] = labels
    
    # Persiapkan informasi cluster
    cluster_info = {
        'medoids': optimal_medoids, 
        'silhouette_score': silhouette, 
        'cluster_sizes': distribution,
        'medoid_rows': df_normalized.iloc[optimal_medoids]
    }
    
    return df_clustered, cluster_info
     
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
    
        if 'df_normalized' not in st.session_state:
            st.warning('Silakan lakukan preprocessing terlebih dahulu')
            return
        
        # Kolom untuk interaksi
        col1, col2 = st.columns(2)
        
        with col1:
            # Slider untuk memilih K
            n_clusters = st.slider('Jumlah Cluster (K)', min_value=2, max_value=6, value=5)
        
        with col2:
            # Tombol untuk analisis
            analyze_button = st.button(f'Analisis dengan K={n_clusters}')
        
        # Tambahkan kolom untuk menampilkan hasil evaluasi
        if analyze_button:
            with st.spinner('Sedang melakukan clustering...'):
                df_clustered, cluster_info = perform_clustering(st.session_state['df_normalized'], n_clusters)
                
                # Visualisasi t-SNE
                st.write("### Visualisasi Cluster")
                plt_tsne = visualize_kmedoids_clusters(df_clustered, cluster_info)
                st.pyplot(plt_tsne)
                
                # Tampilkan perbandingan silhouette score
                if 'clustering_results' not in st.session_state:
                    st.session_state['clustering_results'] = {}
                
                st.session_state['clustering_results'][n_clusters] = {
                    'silhouette': cluster_info['silhouette_score'],
                    'distribution': cluster_info['cluster_sizes']
                }
                
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
        
main()
