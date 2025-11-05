from flask import Flask, render_template, request, send_from_directory, Response
import joblib
import numpy as np
import pandas as pd 

app = Flask(__name__)


try:
    pipeline = joblib.load('model/naive_bayes_pipeline.pkl')
    encoder = pipeline.named_steps['encoder']
    
except FileNotFoundError:
    print("Error: File pipeline 'naive_bayes_pipeline.pkl' tidak ditemukan. Jalankan train_model_final.py terlebih dahulu.")
    exit()
    
@app.route('/data/<path:filename>')
def view_dataset(filename):
    file_path = f"data/{filename}"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    return Response(data, mimetype='text/plain')

@app.route('/')
def landing():
    """Landing page - homepage"""
    return render_template('landing.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    prediksi = None
    probabilitas = None
    input_kriteria = None 

    if request.method == 'POST':
        # 1. Ambil Input dari Formulir
        data_input = {
            'Riwayat_Kredit': request.form['riwayat_kredit'],
            'Lama_Usaha': request.form['lama_usaha'],
            'Pendapatan_Bulan': request.form['pendapatan'],
            'Jaminan': request.form['jaminan'],
            'Jumlah_Pinjaman': request.form['jumlah_pinjaman']
        }
        
        # Simpan input kriteria untuk ditampilkan di HTML
        input_kriteria = data_input 
        input_values = list(data_input.values())
        
        
        fitur = ['Riwayat_Kredit', 'Lama_Usaha', 'Pendapatan_Bulan', 'Jaminan', 'Jumlah_Pinjaman']
        
        # 2. Ubah Input menjadi Pandas DataFrame sementara
    
        input_df = pd.DataFrame([input_values], columns=fitur)
        
        # 3. Prediksi dan Probabilitas menggunakan DataFrame
        prediksi_kelas = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]
        
        
        
        # 4. Hitung Probabilitas
        classes = pipeline.named_steps['classifier'].classes_
        idx_terima = np.where(classes == 'Terima')[0][0]
        idx_tolak = np.where(classes == 'Tolak')[0][0]
        
        prob_terima = proba[idx_terima] * 100
        prob_tolak = proba[idx_tolak] * 100
        
        prediksi = prediksi_kelas
        probabilitas = {
            'Terima': f"{prob_terima:.2f}%",
            'Tolak': f"{prob_tolak:.2f}%"
        }

    
    return render_template(
        'index.html', 
        prediksi=prediksi, 
        probabilitas=probabilitas,
        input_kriteria=input_kriteria,
        # Daftar opsi untuk digunakan di HTML
        riwayat_opts=encoder.categories_[0],
        lama_opts=encoder.categories_[1],
        pendapatan_opts=encoder.categories_[2],
        jaminan_opts=encoder.categories_[3],
        pinjaman_opts=encoder.categories_[4]
    )

if __name__ == '__main__':
    app.run(debug=True)