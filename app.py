from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)



try:
    pipeline = joblib.load('model/naive_bayes_pipeline.pkl')
    encoder = pipeline.named_steps['encoder']
    
except FileNotFoundError:
    print("Error: File pipeline 'naive_bayes_pipeline.pkl' tidak ditemukan. Jalankan train_model_final.py terlebih dahulu.")
    exit()

@app.route('/')
def landing():
    """Landing page - homepage"""
    return render_template('landing.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    prediksi = None
    probabilitas = None
    # --- VARIABEL BARU UNTUK MENYIMPAN INPUT KRITERIA ---
    input_kriteria = None 

    if request.method == 'POST':
        
        data_input = {
            'riwayat_kredit': request.form['riwayat_kredit'],
            'lama_usaha': request.form['lama_usaha'],
            'pendapatan': request.form['pendapatan'],
            'jaminan': request.form['jaminan'],
            'jumlah_pinjaman': request.form['jumlah_pinjaman']
        }
        
        
        input_kriteria = data_input 
        
        # Ubah data input menjadi array untuk prediksi
        input_values = list(data_input.values())
        input_array = np.array([input_values])
        
        # 2. Prediksi dan Probabilitas
        prediksi_kelas = pipeline.predict(input_array)[0]
        proba = pipeline.predict_proba(input_array)[0]
        
        # 3. Hitung Probabilitas
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
        
        riwayat_opts=encoder.categories_[0],
        lama_opts=encoder.categories_[1],
        pendapatan_opts=encoder.categories_[2],
        jaminan_opts=encoder.categories_[3],
        pinjaman_opts=encoder.categories_[4]
    )

if __name__ == '__main__':
    app.run(debug=True)