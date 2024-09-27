import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import tkinter as tk
from tkinter import messagebox

# Memuat model dan tokenizer yang telah disimpan
model_path = "./saved_model"  # Sesuaikan dengan lokasi tempat kamu menyimpan model
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pindahkan model ke GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

# Label mapping: 0 = Negatif, 1 = Netral, 2 = Positif
label_dict = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

# Daftar balasan positif
positive_responses = [
    "Tolong perhatikan ucapanmu, kalau tidak ingin di banned!",
    "Kocak udah diperingatin jangan toxic!",
    "Kamu kena banned seminggu!",
    "Jadilah pribadi yang lebih baik sampai kapan mau toxic!",
    "Bermainlah dengan positif, tanpa menghina ras dan suku!"
]

# Fungsi untuk memberikan balasan positif jika komentar negatif
def respond_to_comment(text):
    # Tokenisasi teks input
    inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Melakukan prediksi
    outputs = loaded_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Ambil label prediksi
    predicted_label = label_dict[predictions.item()]
    
    # Jika prediksi negatif, berikan balasan positif
    if predicted_label == 'Negatif':
        response = random.choice(positive_responses)
    else:
        response = "Komentar Positif."
    
    return predicted_label, response

# Fungsi untuk menangani input dari GUI
def handle_input():
    input_text = entry.get()
    
    if input_text.strip() == "":
        messagebox.showwarning("Input Kosong", "Silakan masukkan komentar!")
        return

    # Proses input dan tampilkan hasil balasan
    predicted_label, response = respond_to_comment(input_text)
    
    # Menambahkan log komentar, klasifikasi, dan balasan ke listbox
    log_text = f"Komentar: {input_text} | Klasifikasi: {predicted_label} | Balasan: {response}"
    log_listbox.insert(tk.END, log_text)

# Membuat jendela Tkinter
window = tk.Tk()
window.title("Better Chat System for Mobile Legends")

# Label dan input untuk komentar
input_label = tk.Label(window, text="Masukkan komentar:")
input_label.pack(pady=5)
entry = tk.Entry(window, width=50)
entry.pack(pady=5)

# Tombol untuk memberikan balasan
submit_button = tk.Button(window, text="Kirim Komentar", command=handle_input)
submit_button.pack(pady=10)

# Listbox untuk menampilkan log komentar, klasifikasi, dan balasan
log_listbox = tk.Listbox(window, width=100, height=10)
log_listbox.pack(pady=10)

# Menjalankan aplikasi Tkinter
window.mainloop()
