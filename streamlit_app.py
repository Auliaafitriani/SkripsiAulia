import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import random
from sklearn.manifold import TSNE

class PSOKMedoids:
    def __init__(self, data, n_clusters, random_state=42, max_iter=100, n_particles=30, w=0.7, c1=1.5, c2=1.5):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        
        # Set seed untuk reproducibilitas
        np.random.seed(self.random_state)
        random.seed(self.random_state)

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

def visualize_kmedoids_clusters(df_clustered, cluster_info, compression_factor=0.13):
    # Ekstrak fitur dan label
    features = df_clustered.drop(columns=['ID', 'PEKERJAAN', 'Cluster']).values
    labels = df_clustered['Cluster'].values
    medoids = cluster_info['medoids']
    medoid_features = features[medoids]

    # Terapkan t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_2d = tsne.fit_transform(features)

    # t-SNE untuk medoids
    tsne_medoids = TSNE(n_components=2, random_state=42, perplexity=min(30, len(medoid_features) - 1))
    medoid_2d = tsne_medoids.fit_transform(medoid_features)

    # Kompresi jarak ke medoids
    for i in range(len(np.unique(labels))):
        mask = labels == i
        cluster_points = features_2d[mask]

        # Hitung vektor dari medoid ke titik
        vectors = cluster_points - medoid_2d[i]

        # Kompresi vektor
        compressed_vectors = vectors * compression_factor

        # Terapkan posisi terkompresi
        features_2d[mask] = medoid_2d[i] + compressed_vectors

    # Buat plot
    plt.figure(figsize=(12, 8))

    # Gunakan warna berbeda
    colors = plt.cm.husl(np.linspace(0, 1, len(np.unique(labels))))

    # Plot titik dan garis
    for i, color in enumerate(colors):
        mask = labels == i
        cluster_points = features_2d[mask]

        # Plot titik
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[color], label=f'Cluster {i}',
                   alpha=0.7, s=100)

        # Gambar garis ke medoid dengan alpha rendah
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

def perform_clustering(df_normalized, n_clusters=5):
    data = df_normalized.drop(columns=['ID', 'PEKERJAAN']).values
    pso = PSOKMedoids(data, n_clusters=n_clusters)
    optimal_medoids, silhouette, labels, distribution = pso.optimize()

    df_clustered = df_normalized.copy()
    df_clustered['Cluster'] = labels
    
    return df_clustered, {
        'medoids': optimal_medoids, 
        'silhouette_score': silhouette, 
        'cluster_sizes': distribution,
        'medoid_rows': df_normalized.iloc[optimal_medoids]
    }

def main():
    st.title('Analisis Clustering Desa')

    # Sidebar untuk unggah file
    st.sidebar.header('Unggah File Excel')
    uploaded_file = st.sidebar.file_uploader("Pilih file Excel", type=['xlsx'])

    if uploaded_file is not None:
        # Baca file Excel
        df = pd.read_excel(uploaded_file)
        st.write("### Data Asli")
        st.dataframe(df)

        # Sidebar untuk konfigurasi clustering
        st.sidebar.header('Konfigurasi Clustering')
        n_clusters = st.sidebar.slider('Jumlah Cluster', min_value=2, max_value=10, value=5)

        # Tombol untuk memulai normalisasi dan clustering
        if st.sidebar.button('Lakukan Normalisasi dan Clustering'):
            # Normalisasi data
            st.write("### Normalisasi Data")
            df_normalized = weighted_normalize(df)
            st.dataframe(df_normalized)

            # Clustering
            st.write("### Hasil Clustering")
            df_clustered, cluster_info = perform_clustering(df_normalized, n_clusters)
            st.dataframe(df_clustered)

            # Tampilkan informasi cluster
            st.write("### Informasi Cluster")
            st.write(f"Silhouette Score: {cluster_info['silhouette_score']:.4f}")
            
            # Distribusi cluster dengan format yang diinginkan
            st.write("### Distribusi Cluster:")
            cluster_distribution = ""
            for i, count in enumerate(cluster_info['cluster_sizes']):
                cluster_distribution += f"Cluster {i}: {count} titik data\n"
            st.text(cluster_distribution)

            # Visualisasi distribusi cluster
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(cluster_info['cluster_sizes'])), cluster_info['cluster_sizes'])
            plt.title('Distribusi Anggota Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Jumlah Anggota')
            st.pyplot(plt)

            # Tampilkan Informasi Medoid Terbaik
            st.write("### Medoid Terbaik")
            st.dataframe(cluster_info['medoid_rows'])

            # Visualisasi Cluster dengan t-SNE
            st.write("### Visualisasi Cluster")
            plt_tsne = visualize_kmedoids_clusters(df_clustered, cluster_info)
            st.pyplot(plt_tsne)

            # Tombol untuk download hasil clustering
            csv = df_clustered.to_csv(index=False)
            st.download_button(
                label="Download Hasil Clustering (CSV)",
                data=csv,
                file_name='hasil_clustering.csv',
                mime='text/csv'
            )

    else:
        st.sidebar.info('Silakan unggah file Excel terlebih dahulu.')

if __name__ == '__main__':
    main()
