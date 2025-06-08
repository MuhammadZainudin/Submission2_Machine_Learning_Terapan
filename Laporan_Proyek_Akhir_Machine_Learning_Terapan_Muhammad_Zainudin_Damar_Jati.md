# Laporan Proyek Machine Learning - Muhammad Zainudin Damar Jati

## Project Overview

Dalam era digital yang dipenuhi jutaan konten hiburan, menemukan anime yang sesuai dengan preferensi pengguna menjadi tantangan tersendiri. Dengan banyaknya pilihan, pengguna membutuhkan sistem yang mampu mempersonalisasi rekomendasi agar waktu eksplorasi lebih efisien. Oleh karena itu, sistem rekomendasi memegang peranan penting dalam menyajikan konten yang relevan, khususnya dalam industri hiburan seperti anime.

Sistem rekomendasi membantu meningkatkan keterlibatan pengguna, memperpanjang waktu tonton, dan mendukung retensi pelanggan. Berdasarkan studi \[Ricci et al., 2015]\[1], sistem rekomendasi terbukti efektif dalam meningkatkan kepuasan pengguna dan mendorong konsumsi konten yang lebih tinggi.

Proyek ini mengembangkan sistem rekomendasi untuk anime dengan menggabungkan dua pendekatan utama: **content-based filtering** dan **collaborative filtering**. Pendekatan ganda ini memungkinkan sistem merekomendasikan anime berdasarkan kemiripan konten (genre) dan juga pola perilaku pengguna lain.

**Referensi**:
\[1] Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.

---

## Business Understanding

### Problem Statements

* Bagaimana memberikan rekomendasi anime yang relevan berdasarkan genre?
* Bagaimana memprediksi preferensi pengguna baru atau lama berdasarkan pola rating pengguna lain?

### Goals

* Menghasilkan daftar rekomendasi anime berdasarkan genre yang mirip dengan anime favorit pengguna.
* Memprediksi rating atau minat pengguna terhadap anime yang belum ditonton, berdasarkan perilaku kolektif pengguna lain.

### Solution Statements

* **Content-Based Filtering**: Menggunakan TF-IDF dan cosine similarity berdasarkan genre.
* **Collaborative Filtering**: Menggunakan matrix factorization (SVD) untuk mempelajari interaksi pengguna–anime dan memprediksi rating.

---

## Data Understanding

Dataset yang digunakan diperoleh dari [Kaggle Anime Recommendation Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) dan terdiri dari dua file utama:

* `anime.csv` (12.294 entri): berisi informasi deskriptif tiap anime.
* `rating.csv` (sekitar 7,8 juta entri): mencatat interaksi pengguna dengan anime melalui rating.

### Fitur-Fitur pada Dataset:

**anime.csv**:

* `anime_id`: ID unik tiap anime.
* `name`: Judul anime.
* `genre`: Daftar genre dalam bentuk string yang dipisahkan koma.
* `type`: Jenis anime (TV, Movie, OVA, dll).
* `episodes`: Jumlah episode.
* `rating`: Rata-rata rating dari semua pengguna.
* `members`: Jumlah pengguna yang menambahkan anime ke daftar.

**rating.csv**:

* `user_id`: ID pengguna.
* `anime_id`: ID anime yang diberi rating.
* `rating`: Nilai rating yang diberikan (1–10), atau -1 jika pengguna belum menilai.

---

### Visualisasi Distribusi Rating Anime

Distribusi rating menunjukkan bagaimana persebaran penilaian terhadap anime secara umum:

![Visualisasi Distribusi Rating Anime](Image/Visualisasi%20Distribusi%20Rating%20Anime.png)

---

### Visualisasi 10 Anime dengan Jumlah Members Terbanyak

Gambar berikut menampilkan 10 anime dengan jumlah anggota (members) terbanyak di database, menunjukkan popularitasnya di kalangan pengguna:

![Visualisasi 10 Anime dengan Jumlah Members Terbanyak](Image/Visualisasi%2010%20Anime%20dengan%20Jumlah%20Members%20Terbanyak.png)

---

### Visualisasi Distribusi Rating oleh Pengguna

Distribusi rating berdasarkan interaksi pengguna terhadap anime:

![Visualisasi Distribusi Rating oleh Pengguna](Image/Visualisasi%20Distribusi%20Rating%20oleh%20Pengguna.png)

---

## Data Preparation

Tahapan persiapan data dilakukan agar data siap untuk modeling:

1. **Menghapus rating tidak valid**:

   * Filter `rating != -1` untuk memastikan hanya interaksi yang valid yang digunakan.

2. **Penanganan missing values**:

   * Genre kosong diisi dengan “Unknown”.
   * Rating kosong pada `anime.csv` diisi dengan median rating.

3. **Normalisasi genre**:

   * Konversi ke huruf kecil (`lowercase`).
   * Hilangkan spasi ekstra.

4. **TF-IDF vectorization**:

   * Genre diolah menjadi representasi numerik menggunakan TF-IDF, dengan token delimiter berupa koma.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
anime['genre'] = anime['genre'].fillna("Unknown").str.lower()
tfidf = TfidfVectorizer(token_pattern=r'[^,]+')
tfidf_matrix = tfidf.fit_transform(anime['genre'])
```

---

## Modeling

### 1. Content-Based Filtering (CBF)

* Menghitung kemiripan antar anime menggunakan cosine similarity pada matriks TF-IDF.
* Output berupa anime dengan genre paling mirip terhadap input favorit pengguna.

```python
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(tfidf_matrix)
```

**Contoh Output Rekomendasi (jika input "Naruto")**:

| Judul Anime       | Genre               | Skor Kemiripan |
| ----------------- | ------------------- | -------------- |
| Naruto: Shippuden | action,adventure    | 1.0            |
| Bleach            | action,supernatural | 0.91           |
| One Piece         | action,comedy       | 0.88           |

**Kelebihan**: Tidak memerlukan data pengguna.
**Kekurangan**: Tidak bisa menangani cold start item (anime tanpa genre).

---

### 2. Collaborative Filtering (SVD)

* Menggunakan algoritma SVD dari library `Surprise` untuk memprediksi rating berdasarkan interaksi pengguna lain.

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(rating_df[['user_id', 'anime_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
```

**Kelebihan**: Dapat memberikan prediksi personal yang akurat.
**Kekurangan**: Tidak dapat merekomendasikan untuk user baru (cold start user).

---

## Evaluation

Evaluasi dilakukan terhadap model collaborative filtering (SVD) menggunakan metrik:

* **MAE (Mean Absolute Error)**:

  $$
  MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
  $$

* **RMSE (Root Mean Squared Error)**:

  $$
  RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
  $$

**Hasil evaluasi SVD**:

* MAE: 0.89
* RMSE: 1.15

---

### Visualisasi Loss Selama Training

Loss function selama proses training model collaborative filtering ditampilkan dalam grafik berikut:

![Visualisasi Loss Selama Training](Image/Visualisasi%20Loss%20Selama%20Training.png)

---

### Visualisasi MAE Selama Training

Mean Absolute Error (MAE) juga divisualisasikan untuk melihat performa model dari waktu ke waktu:

![Visualisasi MAE Selama Training](Image/Visualisasi%20MAE%20Selama%20Training.png)

---

### Visualisasi Scatter Plot Prediksi vs Aktual

Scatter plot berikut menunjukkan hubungan antara rating prediksi dan aktual, membantu kita mengevaluasi ketepatan model:

![Visualisasi Scatter Plot Prediksi vs Aktual](Image/Visualisasi%20Scatter%20Plot%20Prediksi%20vs%20Aktual.png)

---

## Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi anime dengan dua pendekatan:

* **Content-Based Filtering**: Efektif memberikan rekomendasi berdasarkan genre.
* **Collaborative Filtering**: Memberikan rekomendasi personal berdasarkan perilaku pengguna lain dengan akurasi memadai.

**Pengembangan ke depan**:

* Menerapkan sistem hybrid (menggabungkan CBF + CF).
* Menambahkan fitur user demografi untuk rekomendasi yang lebih kontekstual.
* Deploy ke web app atau API menggunakan Flask/Streamlit.
* Eksperimen dengan deep learning (Autoencoder, NCF) untuk performa lebih baik.
