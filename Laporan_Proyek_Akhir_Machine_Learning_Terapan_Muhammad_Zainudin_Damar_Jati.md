# Laporan Proyek Machine Learning - Muhammad Zainudin Damar Jati

## Project Overview

Sistem rekomendasi telah menjadi komponen penting dalam banyak platform digital modern, termasuk dalam industri hiburan seperti streaming anime. Seiring pertumbuhan industri anime dan diversifikasi preferensi penonton, kebutuhan akan sistem yang dapat memberikan rekomendasi personalisasi semakin mendesak. Sistem rekomendasi memungkinkan pengguna untuk menemukan konten baru yang relevan berdasarkan preferensi mereka, sehingga meningkatkan keterlibatan pengguna dan durasi penggunaan platform.

Masalah ini penting diselesaikan karena jumlah anime yang sangat banyak bisa membuat pengguna kesulitan menentukan pilihan tontonan. Pendekatan tradisional seperti pencarian manual atau daftar populer tidak mampu memenuhi kebutuhan personalisasi pengguna secara optimal.

Berdasarkan studi oleh Bobadilla et al. (2013), sistem rekomendasi dapat dikategorikan menjadi Collaborative Filtering, Content-Based Filtering, dan Hybrid Systems, masing-masing dengan kelebihan dan keterbatasan tertentu \[1].

> \[1] J. Bobadilla, F. Ortega, A. Hernando, and A. Gutiérrez, “Recommender systems survey,” *Knowledge-Based Systems*, vol. 46, pp. 109–132, 2013.

---

## Business Understanding

### Problem Statements

* Bagaimana memberikan rekomendasi anime yang relevan berdasarkan genre?
* Bagaimana memprediksi preferensi pengguna baru atau lama berdasarkan pola rating pengguna lain?

### Goals

* Menghasilkan daftar rekomendasi anime berdasarkan genre yang mirip dengan anime favorit pengguna.
* Memprediksi rating atau minat pengguna terhadap anime yang belum ditonton, berdasarkan perilaku kolektif pengguna lain.

### Solution Statements

* **Content-Based Filtering (CBF)** : Sistem rekomendasi berdasarkan kemiripan item. Fitur utama yang digunakan adalah genre anime. Pendekatan ini sangat efektif untuk pengguna baru (cold-start) karena hanya mengandalkan informasi item.
* **Neural Collaborative Filtering (NCF)** : Menggunakan pendekatan berbasis deep learning untuk mempelajari interaksi antara pengguna dan item. Model ini mampu menangkap hubungan non-linear yang kompleks dalam preferensi pengguna.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari [Anime Recommendation Database di Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database), yang terdiri atas dua file utama:

* **`anime.csv`**: berisi metadata tentang anime seperti judul, genre, jenis (TV, Movie, OVA, dll), jumlah episode, rating, dan jumlah anggota komunitas.
* **`rating.csv`**: memuat interaksi pengguna dengan anime berupa pemberian rating, yang mencakup ID pengguna, ID anime, dan nilai rating.

### Library yang Digunakan

Beberapa pustaka yang digunakan dalam proses ini mencakup:

* **Pandas** dan **NumPy**: untuk manipulasi dan analisis data.
* **Matplotlib** dan **Seaborn**: untuk visualisasi eksploratif.
* **Scikit-learn**: untuk preprocessing, evaluasi model, dan teknik representasi teks.
* **TensorFlow/Keras**: untuk membangun model deep learning berbasis neural network.

### Eksplorasi Data

Setelah dataset dimuat, dilakukan eksplorasi awal untuk memahami struktur dan kualitas data. Beberapa temuan penting antara lain:

* Dataset `anime.csv` terdiri dari **12.294** entri dengan 7 kolom.

  * Terdapat nilai kosong (missing values) pada kolom `genre` dan `rating`.
  * Genre disimpan dalam format string yang mengandung beberapa label yang perlu diproses lebih lanjut.
  * Nilai `members` bervariasi sangat besar dan dapat digunakan sebagai indikator popularitas.

* Dataset `rating.csv` terdiri dari lebih dari **7,8 juta** entri.

  * Kolom `rating` berisi nilai dari 1 hingga 10, dengan **nilai -1** menandakan pengguna belum memberikan rating sebenarnya (hanya menonton).
  * Perlu dilakukan filter agar model hanya menggunakan interaksi dengan rating valid (≥1).

### Sampling

Karena ukuran dataset `rating.csv` cukup besar, dilakukan pengambilan sampel sebanyak **50.000 baris** untuk keperluan eksplorasi awal dan efisiensi waktu pemrosesan. Analisis selanjutnya seperti pemeriksaan distribusi rating, jumlah interaksi per pengguna, dan popularitas anime akan dilakukan berdasarkan data sampel ini.

### Visualisasi Distribusi Rating Anime

![Visualisasi Distribusi Rating Anime](imeges/Visualisasi%20Distribusi%20Rating%20Anime.png)

Visualisasi Distribusi Rating Anime ini menunjukkan distribusi rating anime berdasarkan data yang ada. Rating anime paling banyak berada di kisaran 6 hingga 7, yang terlihat dari batang histogram tertinggi pada rentang tersebut. Secara umum, bentuk distribusinya menyerupai lonceng (distribusi normal), yang berarti sebagian besar anime memiliki rating di tingkat menengah, sementara yang memiliki rating sangat rendah atau sangat tinggi jumlahnya lebih sedikit. Garis lengkung biru di atas histogram merupakan kurva KDE (Kernel Density Estimation) yang membantu memperjelas pola sebaran data secara halus. Visualisasi ini memberikan gambaran bahwa rating anime cenderung terpusat di nilai tengah dan jarang ada yang memiliki rating ekstrem.



### Visualisasi 10 Anime dengan Jumlah Members Terbanyak

![Visualisasi 10 Anime dengan Jumlah Members Terbanyak](imeges/Visualisasi%2010%20Anime%20dengan%20Jumlah%20Members%20Terbanyak.png)

Visualisasi 10 Anime dengan Jumlah Members Terbanyak ini menunjukkan 10 anime dengan jumlah members terbanyak berdasarkan data yang tersedia. Pada sumbu horizontal (x) ditampilkan jumlah members, yaitu jumlah pengguna yang menambahkan anime tersebut ke daftar mereka (biasanya di platform seperti MyAnimeList). Sedangkan sumbu vertikal (y) menunjukkan nama-nama anime.

Anime dengan jumlah members terbanyak adalah "Death Note", diikuti oleh "Shingeki no Kyojin" dan "Sword Art Online". Ini menunjukkan bahwa anime-anime tersebut sangat populer dan memiliki banyak penggemar atau penonton yang tertarik untuk menontonnya atau sudah menontonnya.

Grafik ini menggunakan warna gradasi dari palet viridis, yang membantu membedakan tiap batang secara visual. Tampilan horizontal memudahkan pembacaan nama-nama anime yang relatif panjang.




### Visualisasi Distribusi Rating oleh Pengguna

![Visualisasi Distribusi Rating oleh Pengguna](imeges/Visualisasi%20Distribusi%20Rating%20oleh%20Pengguna.png)

Visualisasi Distribusi Rating oleh Pengguna ini menunjukkan distribusi rating yang diberikan oleh pengguna terhadap anime berdasarkan sampel data sebanyak 50.000 entri. Sumbu horizontal (x) menunjukkan nilai rating (dari 1 hingga 10, dan ada satu nilai -1), sedangkan sumbu vertikal (y) menunjukkan jumlah pengguna yang memberikan rating tersebut.

Dari grafik terlihat bahwa:

* Rating 8 adalah yang paling sering diberikan oleh pengguna, diikuti oleh rating 7 dan 9, menandakan bahwa pengguna cenderung memberikan penilaian yang tinggi terhadap anime yang mereka tonton.
* Rating -1 juga memiliki frekuensi yang sangat tinggi. Biasanya nilai ini menunjukkan entri yang belum diberi rating secara eksplisit oleh pengguna dalam data mentah.
* Rating yang lebih rendah seperti 1 hingga 5 jarang diberikan, yang mengindikasikan bahwa pengguna cenderung tidak terlalu sering memberi penilaian buruk.



---

## Data Preparation

### 1. Data Cleaning

#### Menghapus Rating Tidak Valid

Beberapa rating pengguna memiliki nilai `-1`, yang menandakan bahwa pengguna belum memberikan penilaian. Nilai ini tidak relevan dan dihapus agar tidak mengganggu analisis.

```python
rating = rating[rating['rating'] != -1]
```

#### Menangani Nilai Kosong

Nilai kosong pada kolom `genre` dan `type` diisi dengan `'unknown'`, sementara nilai kosong pada kolom `rating` diisi dengan nilai **median** agar tetap representatif terhadap distribusi data.

#### Konversi dan Imputasi Kolom Episodes

Kolom `episodes` awalnya berbentuk teks dan mengandung nilai tak valid seperti `'Unknown'`. Proses konversi dilakukan untuk menjadikannya numerik:

* Nilai tak valid dikonversi menjadi NaN.
* Nilai NaN diisi dengan median.
* Diubah ke integer agar konsisten.

#### Validasi Kolom Penting

Setelah proses pembersihan, dipastikan tidak ada lagi nilai kosong pada kolom `episodes` dan `rating` agar proses model tidak gagal.

```python
assert not anime[['episodes', 'rating']].isnull().any().any()
```

#### Normalisasi Teks

Untuk menjaga konsistensi, seluruh isi kolom teks diubah menjadi huruf kecil dan dihapus spasi yang tidak diperlukan. Ini penting agar data tidak dianggap berbeda hanya karena perbedaan kapitalisasi atau spasi.

#### Perapihan Format Genre

Spasi di sekitar tanda koma dalam kolom `genre` dihapus, agar genre dapat dikenali sebagai token tunggal dalam proses tokenisasi.

#### Menghapus Genre Tidak Valid

Baris dengan genre `'unknown'` dihapus agar hanya data dengan informasi genre yang valid digunakan dalam sistem rekomendasi.


### 2. Membuat Dataset untuk Sistem Rekomendasi

Dataset akhir untuk sistem rekomendasi berbasis konten disiapkan dengan hanya mengambil kolom:

* `anime_id`
* `title`
* `genres`

Kolom-kolom ini disalin dari dataset hasil pembersihan dan dinamai ulang agar lebih deskriptif. Dataset ini akan digunakan untuk ekstraksi fitur dan pembuatan vektor kesamaan.

```python
content_based = anime_cleaned[['anime_id', 'name', 'genre']].copy()
content_based.columns = ['anime_id', 'title', 'genres']
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

### 2. Collaborative Filtering (Neural Collaborative Filtering)

* Menggunakan pendekatan **Neural Collaborative Filtering (NCF)** berbasis deep learning dengan representasi pengguna dan anime dalam bentuk **embedding**.
* Model dilatih untuk memprediksi rating antara pengguna dan anime berdasarkan pola interaksi historis.

```python
# User & Anime Embedding Input
user_input = Input(shape=(1,), name='user_input')
anime_input = Input(shape=(1,), name='anime_input')

# Embedding layer
user_embed = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
anime_embed = Embedding(input_dim=num_anime, output_dim=embedding_dim)(anime_input)

# Flatten & Concatenate
user_vec = Flatten()(user_embed)
anime_vec = Flatten()(anime_embed)
concat = Concatenate()([user_vec, anime_vec])

# Dense layers
x = Dense(128)(concat)
x = LeakyReLU()(x)
x = Dropout(0.4)(x)
x = Dense(64)(x)
x = LeakyReLU()(x)
x = Dropout(0.3)(x)
x = Dense(32)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)

# Output layer
output = Dense(1)(x)

model = Model(inputs=[user_input, anime_input], outputs=output)
model.compile(optimizer=Adam(1e-4), loss=Huber(), metrics=['mae'])
model.fit([x_train[:, 0], x_train[:, 1]], y_train, ...)
```

**Kelebihan**:

* Dapat memodelkan interaksi kompleks antar pengguna dan anime.
* Akurasi lebih tinggi dibanding CF klasik seperti SVD jika data cukup besar.

**Kekurangan**:

* Memerlukan lebih banyak data dan sumber daya komputasi.
* Tidak cocok untuk user baru tanpa histori (cold start user).

---

## Evaluasi Model

### Prediksi dan Transformasi Skala

Setelah proses pelatihan selesai, model digunakan untuk memprediksi rating berdasarkan data validasi. Karena pada tahap preprocessing nilai rating telah dinormalisasi (misalnya ke skala 0–1), maka hasil prediksi perlu dikembalikan ke skala aslinya (1–10) menggunakan fungsi *inverse transform*.

Proses evaluasi dilakukan dengan langkah-langkah berikut:

* Model memprediksi rating berdasarkan pasangan (user, anime).
* Nilai prediksi dan nilai aktual dikembalikan ke skala aslinya.
* Hasil prediksi diubah ke bentuk satu dimensi agar mudah dibandingkan dan divisualisasikan.


### Evaluasi Kinerja Model (MSE, RMSE, MAE)

Untuk menilai akurasi prediksi model sistem rekomendasi, digunakan tiga metrik evaluasi utama: **MSE**, **RMSE**, dan **MAE**. Ketiganya mengukur seberapa jauh hasil prediksi model dari nilai sebenarnya, namun dengan pendekatan yang berbeda.


#### Mean Squared Error (MSE)

**Definisi:**
MSE menghitung rata-rata dari **kuadrat selisih** antara nilai prediksi dan nilai aktual. Metrik ini sensitif terhadap outlier karena memberikan penalti lebih besar untuk kesalahan yang besar.

**Rumus:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$


#### Root Mean Squared Error (RMSE)

**Definisi:**
RMSE adalah **akar kuadrat dari MSE**, sehingga memiliki satuan yang sama dengan skala rating (misalnya 1–10). RMSE memberikan informasi yang lebih mudah dipahami secara praktis dan tetap sensitif terhadap outlier.

**Rumus:**

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$


#### Mean Absolute Error (MAE)

**Definisi:**
MAE menghitung rata-rata dari **nilai absolut selisih** antara nilai aktual dan nilai prediksi. Berbeda dari MSE dan RMSE, MAE **tidak terlalu dipengaruhi oleh outlier**, sehingga memberikan gambaran kesalahan umum secara langsung.

**Rumus:**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$


Hasil evaluasi menunjukkan:

* MSE: 1.4011
* RMSE: 1.1837
* MAE: 0.8966

Artinya, rata-rata kesalahan prediksi model berada di bawah 1 poin rating, yang menandakan bahwa model cukup akurat dalam memprediksi rating pengguna terhadap anime.


### Visualisasi Loss Selama Training

![Visualisasi Loss Selama Training](imeges/Visualisasi%20Loss%20Selama%20Training.png)

Visualisasi Loss Selama Training ini menunjukkan perubahan nilai loss selama proses pelatihan model (training) dari epoch ke epoch. Garis biru mewakili train loss (kesalahan pada data latih), dan garis oranye menunjukkan val loss (kesalahan pada data validasi).

Terlihat bahwa pada awal pelatihan (epoch 0 ke 1), nilai train loss turun drastis, lalu mulai stabil setelahnya. Nilai val loss juga cukup stabil dan rendah sejak awal. Ini menandakan bahwa model belajar dengan cepat di awal dan kemudian mampu menjaga performa yang baik tanpa overfitting. Karena nilai train loss dan val loss sama-sama kecil dan mendekati, ini menunjukkan bahwa model bekerja dengan baik dan konsisten pada data pelatihan maupun data validasi.


### Visualisasi MAE Selama Training

![Visualisasi MAE Selama Training](imeges/Visualisasi%20MAE%20Selama%20Training.png)

Visualisasi MAE Selama Training ini menunjukkan perubahan nilai MAE (Mean Absolute Error) selama proses pelatihan model. Garis biru menunjukkan train MAE (kesalahan rata-rata pada data latih), sedangkan garis oranye menunjukkan val MAE (kesalahan rata-rata pada data validasi).

Dari grafik terlihat bahwa MAE menurun tajam di awal (dari epoch 0 ke 1), lalu terus menurun secara perlahan dan stabil seiring bertambahnya epoch. Nilai train MAE dan val MAE saling berdekatan dan sama-sama rendah, yang berarti model belajar dengan baik dan konsisten, serta tidak mengalami overfitting. Kesalahan prediksi model semakin kecil dari waktu ke waktu, baik pada data latih maupun validasi. Ini menandakan bahwa model cukup akurat dan dapat diandalkan.


### Visualisasi Scatter Plot Prediksi vs Aktual

![Visualisasi Scatter Plot Prediksi vs Aktual](imeges/Visualisasi%20Scatter%20Plot%20Prediksi%20vs%20Aktual.png)

Visualisasi Actual vs Predicted Ratings ini menunjukkan seberapa akurat model dalam memprediksi rating dibandingkan dengan nilai rating yang sebenarnya. Setiap titik biru pada grafik mewakili satu data, misalnya satu ulasan dari pengguna, dengan sumbu horizontal menunjukkan rating yang sebenarnya (Actual Rating), dan sumbu vertikal menunjukkan rating yang diprediksi oleh model (Predicted Rating).

Garis merah putus-putus adalah garis prediksi sempurna, di mana nilai prediksi sama persis dengan nilai aktual. Jika model sangat akurat, maka semua titik akan berada di sepanjang garis ini. Namun, dari grafik terlihat bahwa banyak titik berada di atas garis merah ketika nilai aktual rendah, artinya model sering memberikan rating yang lebih tinggi dari sebenarnya (overestimate). Sebaliknya, saat nilai aktual tinggi, banyak titik berada di bawah garis, menunjukkan model sering memprediksi lebih rendah dari nilai sebenarnya (underestimate).

Sebagai contoh, ketika nilai rating sebenarnya adalah 2, banyak prediksi model mendekati 7 atau lebih, dan ketika rating sebenarnya adalah 9 atau 10, model kadang memprediksi hanya 6 atau 7. Ini menunjukkan bahwa model belum cukup akurat dan cenderung bias ke arah nilai tengah. Untuk meningkatkan akurasi, model dapat ditingkatkan dengan teknik tuning atau menggunakan data pelatihan yang lebih seimbang.


---

## Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi anime yang mampu menyajikan hasil yang relevan dan personal melalui kombinasi pendekatan berbasis konten dan pembelajaran interaksi pengguna. Sistem ini secara efektif mengidentifikasi kesamaan genre untuk menyarankan anime serupa, sekaligus mempelajari pola preferensi pengguna dari histori rating untuk menyesuaikan rekomendasi secara individual.

Seluruh tahap pengembangan, mulai dari pembersihan data, pengolahan atribut genre, pemrosesan nilai kosong, hingga pembentukan representasi embedding dan pelatihan model neural, dilakukan secara sistematis dan menyeluruh. Hasil yang diperoleh menunjukkan bahwa sistem mampu menangkap hubungan antara pengguna dan anime dengan cukup baik.

Lebih jauh, pendekatan berbasis genre memungkinkan sistem memberikan rekomendasi awal yang sesuai bahkan bagi pengguna baru, sementara teknik pembelajaran interaksi memungkinkan personalisasi yang lebih dalam untuk pengguna aktif. Hal ini menunjukkan bahwa solusi yang dibangun mampu menjawab tantangan dalam menyajikan rekomendasi yang relevan secara konten maupun perilaku.

Dengan rancangan yang fleksibel dan pendekatan yang terintegrasi, sistem ini dapat menjadi fondasi yang kuat bagi pengembangan layanan rekomendasi anime yang lebih dinamis dan user-centric ke depannya.
