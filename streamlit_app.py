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
        
        # Display original data
        st.write("### Data Asli")
        st.dataframe(st.session_state['original_data'])
        
        # Check Missing Values
        st.write("### Pengecekan Missing Values")
        missing_values = st.session_state['original_data'].isnull().sum()
        missing_df = pd.DataFrame({
            'Kolom': missing_values.index,
            'Jumlah Missing Values': missing_values.values
        })
        st.dataframe(missing_df)
        
        # Check Outliers
        st.write("### Pengecekan Outliers")
        df = st.session_state['original_data']
        
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
        
        # Tampilkan statistik deskriptif
        st.write("### Statistik Deskriptif")
        st.dataframe(df[selected_columns].describe())
            
        if st.button('Lakukan Preprocessing'):
            try:
                df_normalized = weighted_normalize(st.session_state['original_data'])
                st.session_state['df_normalized'] = df_normalized
                
                st.write("### Data Setelah Preprocessing")
                st.dataframe(df_normalized)
                st.success('Preprocessing selesai!')
            except Exception as e:
                st.error(f'Error saat preprocessing: {str(e)}')

    elif selected == 'PSO and K-Medoids Results':
        st.title('PSO and K-Medoids Analysis')
        
        if 'df_normalized' not in st.session_state:
            st.warning('Silakan lakukan preprocessing terlebih dahulu')
            return
            
        n_clusters = st.slider('Jumlah Cluster', min_value=2, max_value=10, value=5)
        
        if st.button('Mulai Analisis PSO K-Medoids'):
            with st.spinner('Sedang melakukan clustering...'):
                df_clustered, cluster_info = perform_clustering(st.session_state['df_normalized'], n_clusters)
                
                st.write("### Hasil Clustering")
                st.dataframe(df_clustered)
                
                st.write(f"Silhouette Score: {cluster_info['silhouette_score']:.4f}")
                
                # Display cluster distribution
                st.write("### Distribusi Cluster:")
                for i, count in enumerate(cluster_info['cluster_sizes']):
                    st.write(f"Cluster {i}: {count} titik data")
                
                # Visualisasi distribusi cluster
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(cluster_info['cluster_sizes'])), cluster_info['cluster_sizes'])
                plt.title('Distribusi Anggota Cluster')
                plt.xlabel('Cluster')
                plt.ylabel('Jumlah Anggota')
                st.pyplot(plt)
                
                # Visualize clusters with t-SNE
                st.write("### Visualisasi Cluster")
                plt_tsne = visualize_kmedoids_clusters(df_clustered, cluster_info)
                st.pyplot(plt_tsne)
                
                # Best medoids information
                st.write("### Medoid Terbaik")
                st.dataframe(cluster_info['medoid_rows'])
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_clustered.to_csv(index=False)
                    st.download_button(
                        label="Download Hasil (CSV)",
                        data=csv,
                        file_name='hasil_clustering.csv',
                        mime='text/csv'
                    )
                
                with col2:
                    # Convert to Excel
                    output = pd.ExcelWriter('hasil_clustering.xlsx', engine='xlsxwriter')
                    df_clustered.to_excel(output, index=False)
                    output.close()
                    
                    with open('hasil_clustering.xlsx', 'rb') as f:
                        excel_data = f.read()
                    
                    st.download_button(
                        label="Download Hasil (Excel)",
                        data=excel_data,
                        file_name='hasil_clustering.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == '__main__':
    main()
