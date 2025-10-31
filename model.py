import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold 
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import numpy as np


try:
    df = pd.read_csv('data/kredit_mikro.csv')
except FileNotFoundError:
    print("Error: File 'data/kredit_mikro.csv' tidak ditemukan. Pastikan Anda telah membuatnya di folder 'data'.")
    exit()

# 2. Definisikan Kategori dan Feature (TETAP SAMA)
kategori_riwayat = ['Buruk', 'Cukup', 'Baik']
kategori_lama = ['Kurang dari 1 Tahun', '1-3 Tahun', 'Lebih dari 3 Tahun']
kategori_pendapatan = ['Rendah', 'Sedang', 'Tinggi']
kategori_jaminan = ['Tidak Ada', 'Ada']
kategori_pinjaman = ['Kecil', 'Sedang', 'Besar']

kategori_encoder = [
    kategori_riwayat,
    kategori_lama,
    kategori_pendapatan,
    kategori_jaminan,
    kategori_pinjaman
]

fitur = ['Riwayat_Kredit', 'Lama_Usaha', 'Pendapatan_Bulan', 'Jaminan', 'Jumlah_Pinjaman']
target = 'Keputusan'

X = df[fitur]
y = df[target]

# --- 3. Membangun Pipeline Model (TETAP SAMA) ---
model_pipeline = Pipeline([
    ('encoder', OrdinalEncoder(categories=kategori_encoder)),
    ('classifier', CategoricalNB())
])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hitung Akurasi menggunakan Cross-Validation
scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring='accuracy')

print(f"Hasil Akurasi tiap Lipatan ({kf.n_splits} Folds): {scores}")
print(f"Akurasi Rata-rata Cross-Validation: {scores.mean():.4f}")
print(f"Deviasi Standar (Stabilitas Model): {scores.std():.4f}")

model_pipeline.fit(X, y)

if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model_pipeline, 'model/naive_bayes_pipeline.pkl')
