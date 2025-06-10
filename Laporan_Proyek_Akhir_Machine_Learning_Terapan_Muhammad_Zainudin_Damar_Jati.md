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

### Dataset `anime.csv`

Dataset ini berisi informasi metadata terkait anime dan terdiri dari **12.294** entri dan 7 fitur. Deskripsi setiap kolom beserta tipe datanya dan status penggunaannya adalah sebagai berikut:

| Kolom      | Tipe Data | Deskripsi                                                                      | Status Penggunaan                                                          |
| ---------- | --------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `anime_id` | `int`     | ID unik untuk setiap anime.                                                    | Digunakan sebagai *primary key* dan untuk relasi antar tabel.              |
| `name`     | `object`  | Judul anime.                                                                   | Tidak digunakan dalam modeling, hanya untuk tampilan hasil rekomendasi.    |
| `genre`    | `object`  | Genre anime dalam format string, dipisahkan koma (misal: "Action, Adventure"). | Digunakan sebagai fitur dalam content-based filtering setelah diproses.    |
| `type`     | `object`  | Jenis media anime, seperti "TV", "Movie", "OVA", dll.                          | Akan dianalisis lebih lanjut untuk potensi feature tambahan.               |
| `episodes` | `object`  | Jumlah episode. Terkadang berisi nilai "Unknown".                              | Tidak digunakan dalam modeling, dapat digunakan untuk analisis deskriptif. |
| `rating`   | `float`   | Rata-rata rating pengguna terhadap anime (skala 0–10).                         | Digunakan untuk filtering anime dengan skor rendah.                        |
| `members`  | `int`     | Jumlah pengguna yang menambahkan anime ke daftar mereka.                       | Digunakan sebagai indikator popularitas.                                   |

> Tidak ada kolom yang dihapus dari dataset ini, namun beberapa kolom tidak digunakan langsung dalam modeling, melainkan hanya untuk keperluan eksplorasi atau tampilan.




### Dataset `rating.csv`

Dataset ini berisi data interaksi antara pengguna dan anime, terdiri dari lebih dari **7,8 juta** entri dan 3 kolom sebagai berikut:

| Kolom      | Tipe Data | Deskripsi                                                                          | Status Penggunaan                                                       |
| ---------- | --------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `user_id`  | `int`     | ID pengguna (anonim).                                                              | Digunakan dalam collaborative filtering.                                |
| `anime_id` | `int`     | ID anime, berfungsi sebagai foreign key untuk join dengan `anime.csv`.             | Digunakan untuk relasi antar tabel.                                     |
| `rating`   | `int`     | Rating yang diberikan oleh pengguna (skala 1–10, -1 berarti belum memberi rating). | Rating -1 akan **dibuang** dari data; sisanya digunakan dalam model CF. |

> Nilai `rating = -1` menandakan pengguna hanya menonton tanpa memberi penilaian eksplisit, sehingga entri tersebut akan **dihapus** dari proses modeling.

### Struktur Relasi Antar Tabel

Berikut adalah struktur relasi sederhana antara kedua dataset:

```
anime.csv (anime_id) ←───────┐
                             │
                      rating.csv
                   (user_id, anime_id, rating)
```

* `anime_id` berfungsi sebagai **primary key** di `anime.csv` dan sebagai **foreign key** di `rating.csv`, menghubungkan informasi metadata anime dengan interaksi pengguna.


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

### A. Content-Based Filtering

#### 1. Pembersihan dan Penyesuaian Data

Langkah awal dalam pendekatan *Content-Based Filtering* adalah memastikan data `anime.csv` dan `rating.csv` bersih dan konsisten. Nilai rating yang tidak valid, seperti `-1`, dihapus karena tidak merepresentasikan penilaian nyata dari pengguna.

Kolom-kolom penting seperti `genre`, `type`, dan `rating` diperiksa dari nilai kosong. Genre dan type yang kosong diisi dengan label `'unknown'`, sementara rating yang kosong diisi dengan nilai **median**, agar tetap mencerminkan distribusi data secara umum.

Kolom `episodes`, yang awalnya bertipe teks dan mengandung nilai tak valid seperti `'Unknown'`, dikonversi menjadi numerik. Nilai-nilai yang tidak valid dikonversi ke `NaN`, diisi dengan nilai **median**, dan diubah ke tipe integer.

Seluruh kolom teks dinormalisasi dengan mengubah huruf menjadi kecil dan menghapus spasi yang tidak diperlukan, termasuk spasi di sekitar tanda koma dalam kolom `genre`. Hal ini dilakukan agar genre dapat dikenali dengan baik saat proses tokenisasi.

Akhirnya, baris-baris dengan genre `'unknown'` dihapus, sehingga hanya data dengan informasi genre yang valid yang digunakan untuk proses rekomendasi.

#### 2. Persiapan Dataset Rekomendasi

Dari dataset yang telah dibersihkan, dibuat subset khusus yang hanya berisi ID anime, judul, dan genre. Nama kolom disesuaikan agar lebih deskriptif dan konsisten.

Genre kemudian diubah menjadi representasi numerik menggunakan metode **TF-IDF Vectorization**, yang memetakan setiap genre ke dalam vektor berdimensi tinggi berdasarkan frekuensi dan keunikan tiap genre. Token genre dipisahkan berdasarkan koma.

Setelah itu, dilakukan perhitungan **cosine similarity** antar vektor anime, yang menghasilkan skor kesamaan antar anime berdasarkan informasi genre. Matriks kesamaan ini menjadi dasar sistem rekomendasi berbasis konten.


### B. Collaborative Filtering

#### 1. Pembersihan Data Rating

Dalam pendekatan *Collaborative Filtering*, fokus utama ada pada data interaksi pengguna dengan anime. Oleh karena itu, hanya data dengan nilai `user_id`, `anime_id`, dan `rating` yang lengkap dan valid (lebih dari nol) yang digunakan.

#### 2. Encoding dan Transformasi

Karena model akan memproses data numerik, ID pengguna dan ID anime dikonversi menjadi indeks numerik melalui proses **label encoding**. Mapping dari ID asli ke indeks disimpan untuk keperluan interpretasi hasil model nantinya.

Nilai rating kemudian dinormalisasi ke dalam skala 0–1 menggunakan **Min-Max Scaler**, agar lebih sesuai dengan fungsi aktivasi dalam model neural collaborative filtering.

#### 3. Pembentukan Dataset

Data interaksi kemudian dipecah menjadi fitur (berisi pasangan user dan anime) dan target (berisi rating). Dataset ini kemudian diacak dan dibagi menjadi dua bagian: data latih (80%) dan data validasi (20%), untuk memastikan evaluasi model lebih representatif.

---

## Modeling

Proses modeling dilakukan menggunakan dua pendekatan utama, yaitu:

1.  **Content-Based Filtering (CBF)** — berbasis konten (*genre*).
2.  **Collaborative Filtering (CF)** — berbasis interaksi pengguna dan *item*, menggunakan arsitektur **Neural Collaborative Filtering (NCF)**.

---

### A. Content-Based Filtering (CBF)

#### Penjelasan Metode:

*Content-Based Filtering* memberikan rekomendasi berdasarkan **kemiripan konten** antar anime. Dalam konteks ini, konten yang digunakan adalah **genre** anime. *Genre* tersebut diolah menggunakan metode **TF-IDF (Term Frequency–Inverse Document Frequency)** untuk membentuk representasi numerik dari setiap anime. Setelah itu, dihitung **cosine similarity** antar vektor *genre* untuk mengukur kemiripan.

#### Proses Implementasi:

1.  **Pengolahan *Genre* dan TF-IDF**: *Genre* setiap anime digabungkan ke dalam satu *string* (misalnya: `"action,adventure,supernatural"`) dan kemudian diubah menjadi vektor numerik menggunakan TF-IDF.
2.  **Perhitungan *Cosine Similarity***: *Cosine similarity* dihitung antara seluruh anime untuk mendapatkan matriks kemiripan.
3.  **Fungsi Rekomendasi (`recommend_anime_content_based`):**
    * Mencari anime berdasarkan judul *input* (dikonversi ke huruf kecil dan di-*trim*).
    * Jika anime tidak ditemukan, fungsi akan mengembalikan `top_n` anime secara acak sebagai *fallback*.
    * Jika ditemukan, indeks anime tersebut diambil, lalu anime lain dengan nilai *cosine similarity* tertinggi akan dicari.
    * Mengembalikan `top_n` anime paling mirip beserta skornya, seperti contoh rekomendasi untuk "Naruto":

        ```
        print(recommend_anime_content_based("Naruto"))
        # Output:
        #                      title                                             genres  similarity_score
        # 615        naruto: shippuuden  action,comedy,martial arts,shounen,super power               1.0
        # 841                    naruto  action,comedy,martial arts,shounen,super power               1.0
        # 1103  boruto: naruto the movie - naruto ga hokage ni...  action,comedy,martial arts,shounen,super power               1.0
        # 1343               naruto x ut  action,comedy,martial arts,shounen,super power               1.0
        # 1472  naruto: shippuuden movie 4 - the lost tower  action,comedy,martial arts,shounen,super power               1.0
        ```

#### Kelebihan:

* Tidak membutuhkan data interaksi pengguna (*suitable for cold-start user*).
* Mudah diterapkan karena hanya bergantung pada metadata (*genre*).

#### Kelemahan:

* Rekomendasi kurang personal (berbasis konten, bukan preferensi pengguna).
* Tidak bisa memberikan rekomendasi jika data *genre* kosong atau tidak informatif.



### B. Collaborative Filtering (Neural Collaborative Filtering)

#### Penjelasan Metode:

Metode ini menggunakan teknik **Neural Collaborative Filtering (NCF)**, yaitu pendekatan *deep learning* untuk memprediksi interaksi antara pengguna dan anime. Model belajar dari pola *rating* atau interaksi pengguna sebelumnya untuk membuat rekomendasi yang bersifat personal.

#### Arsitektur Model:

Model NCF dibangun dengan arsitektur *embedding* yang memetakan ID pengguna dan ID anime ke dalam ruang vektor berdimensi tetap. Arsitektur ini meniru pendekatan *matrix factorization* dengan fleksibilitas *deep learning*. Berikut adalah detail lapisannya:

* **Input**: `user_input` (ID pengguna) dan `anime_input` (ID anime).
* **Embedding Layer**: Masing-masing ID dikonversi menjadi vektor *embedding* berdimensi tetap (misalnya 64). Regularisasi `l2(1e-5)` diterapkan pada *embedding* untuk mencegah *overfitting*.
* **Flatten Layer**: Mengubah *embedding* dari dimensi (None, 1, 64) menjadi (None, 64).
* **Concatenate Layer**: Menggabungkan vektor *embedding* pengguna dan anime.
* **Hidden Layers**: Gabungan *embedding* dilewatkan melalui beberapa lapisan *dense neural network* dengan aktivasi **LeakyReLU** dan **Batch Normalization**. *Dropout* (`0.4`, `0.3`, `0.2`) dan regularisasi `l2(1e-4)` digunakan untuk mencegah *overfitting*.
    * Dense 1: 128 *unit*
    * Dense 2: 64 *unit*
    * Dense 3: 32 *unit*
* **Output Layer**: Lapisan *dense* dengan 1 *unit* untuk memprediksi skor *rating* atau kemungkinan interaksi.

#### Proses *Training*:

1.  **Pembagian Data**: Data dibagi menjadi data pelatihan (`x_train`, `y_train`) dan data validasi (`x_val`, `y_val`).
2.  **Fungsi *Loss***: Model dioptimasi menggunakan **Adam Optimizer** dengan *learning rate* awal `1e-4` dan metrik **MAE (Mean Absolute Error)**. Fungsi *loss* yang digunakan adalah `weighted_huber`, sebuah modifikasi dari *Huber Loss* dengan pembobotan tambahan. *Rating* di bawah *threshold* (`0.3`) akan memiliki bobot lebih besar (`alpha=2.0`), berguna jika ingin lebih sensitif terhadap *rating* rendah (misalnya *outlier* atau ketidakpuasan pengguna).
3.  ***Callbacks***:
    * **Early Stopping**: `early_stop` menghentikan pelatihan jika `val_loss` tidak membaik selama 5 *epoch*, dan akan mengembalikan *weights* terbaik.
    * **Learning Rate Scheduler**: `reduce_lr` akan menurunkan *learning rate* (faktor `0.5`, minimum `1e-6`) jika `val_loss` stagnan selama 3 *epoch*.
    Kedua *callback* ini meningkatkan efisiensi dan performa pelatihan serta mencegah *overfitting*.
4.  **Pelatihan Model**: Model dilatih selama maksimal 50 *epoch* dengan *batch size* 128. *Callbacks* yang telah didefinisikan digunakan untuk mengatur pelatihan secara dinamis.

#### Kelebihan:

* Memberikan rekomendasi yang lebih personal karena mempelajari preferensi pengguna.
* Mampu menangkap hubungan non-linear dan kompleks antara pengguna dan anime.
* Lebih akurat daripada model CF klasik seperti SVD, terutama dengan jumlah data yang besar.

#### Kekurangan:

* Membutuhkan data interaksi historis pengguna (tidak cocok untuk pengguna baru atau *cold-start user*).
* Butuh waktu pelatihan dan komputasi yang lebih besar.
* Model kompleks dan sulit diinterpretasi secara langsung.

---

## Evaluasi Model

### A. Evaluasi Model Content-Based Filtering

Model *Content-Based Filtering* merekomendasikan item berdasarkan atribut atau fitur dari item itu sendiri yang disukai pengguna di masa lalu. Dalam kasus ini, rekomendasi anime didasarkan pada kesamaan genre. Untuk menilai relevansi rekomendasi, kami menggunakan metrik **Precision@5** dan **Recall@5**.

#### Precision@5

**Precision@5** mengukur proporsi rekomendasi teratas (dalam hal ini, 5 rekomendasi teratas) yang relevan. Relevansi di sini didefinisikan sebagai rekomendasi anime yang memiliki setidaknya satu genre yang sama dengan genre anime asli yang menjadi dasar rekomendasi.

#### Recall@5

**Recall@5** mengukur proporsi genre relevan yang berhasil direkomendasikan oleh model dari total genre yang ada pada anime asli. Metrik ini membantu melihat seberapa lengkap cakupan rekomendasi dalam menangkap preferensi genre pengguna.

#### Hasil Evaluasi Model Content-Based Filtering

Setelah melakukan evaluasi dengan 100 sampel anime, model *Content-Based Filtering* menunjukkan hasil sebagai berikut:

  * **Precision@5: 1.00**
  * **Recall@5: 0.97**

Hasil ini menunjukkan bahwa model *Content-Based Filtering* **sangat efektif** dalam merekomendasikan anime dengan genre yang relevan. Nilai Precision@5 sebesar 1.00 berarti bahwa semua (100%) dari 5 rekomendasi teratas yang diberikan oleh model memiliki setidaknya satu genre yang tumpang tindih dengan anime asli. Sementara itu, Recall@5 sebesar 0.97 menunjukkan bahwa hampir semua genre dari anime asli berhasil tercakup dalam daftar rekomendasi. Ini mengindikasikan bahwa model tidak hanya merekomendasikan anime yang relevan, tetapi juga cukup komprehensif dalam mencakup preferensi genre pengguna.

### B. Evaluasi Model Collaborative Filtering

Model *Collaborative Filtering* bekerja dengan menemukan pola dalam interaksi pengguna-item (misalnya, rating yang diberikan pengguna kepada anime) untuk membuat rekomendasi. Untuk model ini, kami mengevaluasi akurasi prediksi rating menggunakan metrik **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, dan **Mean Absolute Error (MAE)**. Sebelum evaluasi, semua prediksi rating dinormalisasi kembali ke skala aslinya (1-10).


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
