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

# Define valid columns globally
VALID_COLUMNS = ['ID', 'PEKERJAAN', 'JUMLAH ASET MOBIL', 'JUMLAH ASET MOTOR', 
                 'JUMLAH ASET RUMAH/TANAH/SAWAH', 'PENDAPATAN']

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

def validate_columns(df):
    """Validate that the DataFrame contains only the allowed columns."""
    current_columns = set(df.columns)
    required_columns = set(VALID_COLUMNS)
    
    if not required_columns.issubset(current_columns):
        missing_columns = required_columns - current_columns
        raise ValueError(f"The data is missing the required columns: {', '.join(missing_columns)}")
    
    extra_columns = current_columns - required_columns
    if extra_columns:
        raise ValueError(f"The data contains additional columns that are not allowed: {', '.join(extra_columns)}")
    
    return True

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

def visualize_kmedoids_clusters(df_clustered, cluster_info, compression_factor=0.05):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    import seaborn as sns

    # Extract features and labels
    features = df_clustered.drop(columns=['ID', 'PEKERJAAN', 'Cluster']).values
    labels = df_clustered['Cluster'].values

    # Get medoids
    medoids = cluster_info.get('medoids', cluster_info.get('medoid_indices', []))
    
    # Pastikan medoids berisi indeks yang valid
    medoids = [m for m in medoids if m < len(features)]
    
    # Hindari error jika medoids kosong
    if not medoids:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Tidak dapat membuat visualisasi', 
                 horizontalalignment='center', 
                 verticalalignment='center')
        return plt

    medoid_features = features[medoids]

    # Apply t-SNE with adjusted perplexity for medoids
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_2d = tsne.fit_transform(features)

    # Use a smaller perplexity for medoids
    tsne_medoids = TSNE(n_components=2, random_state=42, perplexity=min(30, len(medoid_features) - 1))
    medoid_2d = tsne_medoids.fit_transform(medoid_features)

    # Compress distances to medoids
    for i in range(len(np.unique(labels))):
        mask = labels == i
        cluster_points = features_2d[mask]

        # Hitung vektor dari medoid ke titik
        if i < len(medoid_2d):
            vectors = cluster_points - medoid_2d[i]

            # Kompresi vektor
            compressed_vectors = vectors * compression_factor

            # Terapkan posisi terkompresi
            features_2d[mask] = medoid_2d[i] + compressed_vectors

    # Plot
    plt.figure(figsize=(12, 8))

    # Gunakan warna berbeda
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    # Plot titik dan garis
    for i, color in enumerate(colors):
        mask = labels == unique_labels[i]
        cluster_points = features_2d[mask]
        
        # Plot titik
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[color], label=f'Cluster {unique_labels[i]}',
                   alpha=0.7, s=100)
        
        # Gambar garis ke medoid dengan alpha rendah
        if i < len(medoid_2d):
            for point in cluster_points:
                plt.plot([medoid_2d[i, 0], point[0]],
                        [medoid_2d[i, 1], point[1]],
                        c=color, alpha=0.2, linewidth=0.5)

    # Plot medoids
    plt.scatter(medoid_2d[:, 0], medoid_2d[:, 1],
               c='red', marker='*', s=800,
               label='Medoids', edgecolor='black', linewidth=2)

    plt.title('Visualisasi Clustering K-Medoids', fontsize=14, pad=20)
    plt.xlabel('Komponen t-SNE 1', fontsize=12)
    plt.ylabel('Komponen t-SNE 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              borderaxespad=0., fontsize=10)
    plt.tight_layout()
    
    return plt

def search_by_id(df_original, df_normalized, cluster_results):
    st.write("### Detailed Data Search")
    
    # Input ID
    search_id = st.text_input('Enter the ID to be searched:')
    
    # Pilih K
    available_k = list(cluster_results.keys())
    selected_k = st.selectbox('Select the Number of Clusters (K):', available_k)
    
    if st.button('search'):
        try:
            # Pastikan search_id adalah integer
            search_id_int = int(search_id)
            
            # Cari data asli berdasarkan ID
            original_data = df_original[df_original['ID'] == search_id_int]
            
            # Cari cluster untuk ID ini di hasil clustering untuk K yang dipilih
            cluster_df = cluster_results[selected_k]['df_clustered']
            cluster = cluster_df[cluster_df['ID'] == search_id_int]['Cluster'].values
            
            if not original_data.empty:
                st.write("### Data Details:")
                st.dataframe(original_data)
                
                if len(cluster) > 0:
                    st.write(f"### Cluster: {cluster[0]}")
                    
                    # Tampilkan detail cluster
                    st.write("### Cluster Information:")
                    cluster_details = cluster_df[cluster_df['Cluster'] == cluster[0]]
                    st.write(f"Jumlah anggota Cluster {cluster[0]}: {len(cluster_details)}")
                else:
                    st.warning("ID not found in the selected cluster.")
            else:
                st.warning("ID not found.")
        
        except ValueError:
            st.error("Please enter a valid ID.")

def main():
    # Sidebar for navigation
    with st.sidebar:
        # Tambahkan logo di bagian atas tanpa margin
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src="https://raw.githubusercontent.com/Auliaafitriani/SkripsiAulia/main/LogoPriorityAid.png" 
                     alt="Logo" 
                     width="250" 
                     style="margin-top: 0;">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Menu navigasi dengan option_menu
        selected = option_menu(None,  # Hapus judul menu
                           ['About', 'Upload Data', 'Preprocessing', 
                            'PSO and K-Medoids Results'],
                           menu_icon='cast',
                           icons=['house', 'cloud-upload', 'gear', 'graph-up'],
                           default_index=0,
                           styles={
                               "container": {
                                   "padding": "0px", 
                                   "background-color": "#f0f2f6"
                               },
                               "icon": {
                                   "color": "#3498db", 
                                   "font-size": "17px"
                               }, 
                               "nav-link": {
                                   "font-size": "15px", 
                                   "text-align": "left", 
                                   "margin":"1px", 
                                   "--hover-color": "#e0e0e0"
                               },
                               "nav-link-selected": {
                                   "background-color": "#3498db", 
                                   "color": "white"
                               },
                           })
        
        # Info versi dengan styling minimal
        st.markdown("""
        <div style="text-align:center; padding:1; background-color:#f0f2f6; border-radius:3px; margin-top:10px;">
            <small style="color:#7f8c8d; font-size:15px;">Versi 1.0</small><br>
            <small style="color:#7f8c8d; font-size:15px;">© 2025 Aulia Nur Fitriani</small>
        </div>
        """, unsafe_allow_html=True)

    if selected == 'About':
        st.title('PriorityAid Analytics Dashboard')
        st.write("""
            PriorityAid Analytics Dashboard adalah platform analisis berbasis data yang dirancang untuk meningkatkan ketepatan distribusi bantuan pemerintah di Desa Kalipuro. 
            Proyek ini mengoptimalkan metode K-Medoids dengan Particle Swarm Optimization (PSO) untuk meningkatkan efisiensi dan akurasi dalam pengelompokan penerima bantuan, memastikan distribusi yang lebih adil dan tepat sasaran. 
            Dengan visualisasi interaktif, pemangku kepentingan dapat dengan mudah memahami pola distribusi, mengevaluasi hasil analisis, serta mengidentifikasi tren penerima bantuan secara lebih transparan dan berbasis data, sehingga proses pengambilan keputusan menjadi lebih efektif.
        """)
      
      # Add Terms & Conditions section
        with st.expander("Terms & Conditions"):
            st.markdown("""
            **Allowed Columns:**
            - ID
            - PEKERJAAN
            - JUMLAH ASET MOBIL
            - JUMLAH ASET MOTOR
            - JUMLAH ASET RUMAH/TANAH/SAWAH
            - PENDAPATAN
            
            The uploaded data must contain only the specified columns.
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

        def handle_outliers_iqr(df, columns):
            """Tangani outlier menggunakan metode IQR."""
            df_copy = df.copy()
            for column in columns:
                # Menghitung kuartil pertama (Q1) dan ketiga (Q3)
                Q1 = df_copy[column].quantile(0.25)
                Q3 = df_copy[column].quantile(0.75)
                IQR = Q3 - Q1  # Rentang Interquartile

                # Menentukan batas bawah dan batas atas
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Mengganti nilai yang berada di luar batas dengan batas terdekat
                df_copy[column] = df_copy[column].apply(
                    lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
                )
            return df_copy

        def handle_missing_values(df):
            """Tangani missing values dengan mean untuk kolom numerik."""
            df_copy = df.copy()
            for column in df_copy.select_dtypes(include=['number']).columns:
                df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
            return df_copy
        
        if 'original_data' not in st.session_state or st.session_state['original_data'] is None:
            st.warning('Silakan upload data terlebih dahulu pada halaman Upload Data')
            return
            
        if st.button('Execute Preprocessing'):
            try:
                df = st.session_state['original_data']

                # 1. Statistik Deskriptif
                st.write("### Statistik Deskriptif")
                numeric_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
                st.dataframe(df[numeric_columns].describe())
                
                # 2. Pengecekan Missing Values
                st.write("### Check for Missing Values")
                missing_values = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Kolom': missing_values.index,
                    'Jumlah Missing Values': missing_values.values
                })
                st.dataframe(missing_df)

                # 3. Tangani Missing Values
                df = handle_missing_values(df)
                st.write("### Data After Handling Missing Values")
                st.dataframe(df)
                
                # 4. Pengecekan Outliers Sebelum Penanganan
                st.write("### Outlier Check Before Handling")
                columns_to_check = ['JUMLAH ASET MOBIL', 'JUMLAH ASET MOTOR', 
                                  'JUMLAH ASET RUMAH/TANAH/SAWAH', 'PENDAPATAN']
                
                # Hitung outliers sebelum penanganan
                Q1 = df[columns_to_check].quantile(0.25)
                Q3 = df[columns_to_check].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((df[columns_to_check] < lower_bound) | 
                                 (df[columns_to_check] > upper_bound)).sum()
                outlier_df_before = pd.DataFrame({
                    'Kolom': outliers_before.index,
                    'Jumlah Outlier': outliers_before.values
                })
                st.dataframe(outlier_df_before)

                # 5. Tangani Outliers
                df = handle_outliers_iqr(df, columns_to_check)
                st.write("### Data After Outlier Handling")
                st.dataframe(df)

                # 6. Pengecekan Outliers Setelah Penanganan
                st.write("### Outlier Check After Handling")
                Q1 = df[columns_to_check].quantile(0.25)
                Q3 = df[columns_to_check].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_after = ((df[columns_to_check] < lower_bound) | 
                                (df[columns_to_check] > upper_bound)).sum()
                outlier_df_after = pd.DataFrame({
                    'Kolom': outliers_after.index,
                    'Jumlah Outlier': outliers_after.values
                })
                st.dataframe(outlier_df_after)
                
                # 7. Normalisasi dan Pembobotan
                st.write("### Data After Normalization and Weighting")
                df_normalized = weighted_normalize(df)
                st.session_state['df_normalized'] = df_normalized
                st.dataframe(df_normalized)
                
                st.success('Preprocessing completed!')
            except Exception as e:
                st.error(f'Error saat preprocessing: {str(e)}')
                
    elif selected == 'PSO and K-Medoids Results':
        st.title('PSO and K-Medoids Analysis')
    
        if 'df_normalized' not in st.session_state:
            st.warning('Please perform preprocessing first.')
            return

        # Inisialisasi dictionary untuk menyimpan semua hasil clustering
        if 'all_clustering_results' not in st.session_state:
            st.session_state['all_clustering_results'] = {}
        
        # Kolom untuk interaksi
        col1, col2 = st.columns(2)
        
        with col1:
            # Slider untuk memilih K
            n_clusters = st.slider('Jumlah Cluster (K)', min_value=2, max_value=5, value=5)
        
        with col2:
            # Tombol untuk analisis
            analyze_button = st.button(f'Analisis dengan K={n_clusters}')
        
        # Tambahkan kolom untuk menampilkan hasil evaluasi
        if analyze_button:
            with st.spinner('Clustering in progress...'):
                df_clustered, cluster_info = perform_clustering(st.session_state['df_normalized'], n_clusters)
                
                # Simpan hasil clustering untuk nilai K ini
                st.session_state['all_clustering_results'][n_clusters] = {
                    'df_clustered': df_clustered,
                    'cluster_info': cluster_info,
                    'silhouette': cluster_info['silhouette_score'],
                    'distribution': cluster_info['cluster_sizes']
                }

        # Tampilkan hasil untuk semua K yang sudah dianalisis
        if st.session_state['all_clustering_results']:
            st.write("## Analysis Results for All K Values")
            
            # Tampilkan tabs untuk setiap nilai K
            tabs = st.tabs([f"K={k}" for k in sorted(st.session_state['all_clustering_results'].keys())])
            
            for i, k in enumerate(sorted(st.session_state['all_clustering_results'].keys())):
                with tabs[i]:
                    results = st.session_state['all_clustering_results'][k]
                    
                    # Visualisasi t-SNE untuk nilai K ini
                    st.write(f"### Cluster Visualization for K={k}")
                    plt_tsne = visualize_kmedoids_clusters(results['df_clustered'], results['cluster_info'])
                    st.pyplot(plt_tsne)
                    
                    # Informasi medoid terbaik
                    st.write(f"### Best Medoid for K={k}")
                    st.dataframe(results['cluster_info']['medoid_rows'])
                    
                    # Distribusi cluster
                    st.write(f"### Cluster Distribution for K={k}")
                    cluster_distribution = f"Distribusi Cluster (K={k}): \n"
                    for j, count in enumerate(results['distribution']):
                        cluster_distribution += f"Cluster {j}: {count} titik data\n"
                    st.text(cluster_distribution)
                    
                    # Menghapus download button dari tab individu
            
            # Tampilkan perbandingan Silhouette Score
            st.write("### Silhouette Score Comparison")
            comparison_data = {
                'K': list(st.session_state['all_clustering_results'].keys()),
                'Silhouette Score': [results['silhouette'] for results in st.session_state['all_clustering_results'].values()]
            }
            comparison_df = pd.DataFrame(comparison_data)
            
            # Tampilkan tabel perbandingan
            st.dataframe(comparison_df)
            
            # Visualisasi perbandingan
            plt.figure(figsize=(10, 6))
            plt.plot(comparison_df['K'], comparison_df['Silhouette Score'], marker='o')
            plt.title('Silhouette Score Comparison for Different K Values')
            plt.xlabel('Jumlah Cluster (K)')
            plt.ylabel('Silhouette Score')
            plt.grid(True)
            st.pyplot(plt)
            
            # Download section
            st.write("### Download Clustering Results")
            
            # Pilih K untuk download
            k_to_download = st.selectbox(
                'Select the K value to download:',
                sorted(st.session_state['all_clustering_results'].keys())
            )
            
            # Tombol download
            if k_to_download is not None:
                csv = st.session_state['all_clustering_results'][k_to_download]['df_clustered'].to_csv(index=False)
                st.download_button(
                    label=f"Download Results for K={k_to_download}",
                    data=csv,
                    file_name=f'hasil_clustering_k{k_to_download}.csv',
                    mime='text/csv',
                    key='download_clustering_results'
                )
            
            # Tambahkan tombol untuk reset hasil
            if st.button('Reset All Analysis Results'):
                st.session_state['all_clustering_results'] = {}
                st.experimental_rerun()

            # Tambahkan pemanggilan search_by_id
            if st.session_state['all_clustering_results']:
                st.write("## Detailed Data Lookup")
                search_by_id(
                    st.session_state['original_data'], 
                    st.session_state['df_normalized'], 
                    st.session_state['all_clustering_results']
                )
      
main()
