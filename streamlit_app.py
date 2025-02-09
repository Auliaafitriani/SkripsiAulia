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
        
        # Tampilkan parameter dan rumus PSO
        st.write("### Parameter PSO yang Digunakan")
        st.write("""
        - w (inertia weight) = 0.7
        - c1 (cognitive coefficient) = 1.5
        - c2 (social coefficient) = 1.5
        - Jumlah Partikel = 30
        - Maksimum Iterasi = 100
        """)
        
        st.write("### Rumus Update Velocity PSO")
        st.latex(r'''
        v_i^{t+1} = w \times v_i^t + c_1r_1(pbest_i^t - x_i^t) + c_2r_2(gbest^t - x_i^t)
        ''')
        st.write("""
        Dimana:
        - v_i^{t+1} : Kecepatan partikel i pada iterasi t+1
        - w : Inertia weight
        - c1 : Cognitive coefficient
        - c2 : Social coefficient
        - r1, r2 : Random number antara 0 dan 1
        - pbest : Personal best position
        - gbest : Global best position
        - x : Current position
        """)
        
        st.write("### Rumus Update Position")
        st.latex(r'''
        x_i^{t+1} = x_i^t + v_i^{t+1}
        ''')
        
        st.write("### Fungsi Objektif")
        st.latex(r'''
        Cost = \sum_{i=1}^n \min_{j=1}^k d(x_i, m_j)
        ''')
        st.write("""
        Dimana:
        - n : Jumlah data points
        - k : Jumlah cluster
        - d(x_i, m_j) : Jarak Euclidean antara data point x_i dan medoid m_j
        """)

        if 'df_normalized' not in st.session_state:
            st.warning('Silakan lakukan preprocessing terlebih dahulu')
            return
            
        # Tambahkan slider untuk memilih K
        n_clusters = st.slider('Jumlah Cluster (K)', min_value=2, max_value=6, value=2)
        
        # Tambahkan kolom untuk menampilkan hasil evaluasi
        if st.button(f'Analisis dengan K={n_clusters}'):
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
        
    if __name__ == '__main__':
            main()
