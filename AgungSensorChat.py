import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import tkinter as tk
from tkinter import messagebox


model_path = "./saved_model"  
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)


label_dict = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}


positive_responses = [
    "Tolong perhatikan ucapanmu, kalau tidak ingin di banned!",
    "Kocak udah diperingatin jangan toxic!",
    "Kamu kena banned seminggu!",
    "Jadilah pribadi yang lebih baik sampai kapan mau toxic!",
    "Bermainlah dengan positif, tanpa menghina ras dan suku!"
]


def respond_to_comment(text):
    
    inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    
    outputs = loaded_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    
    predicted_label = label_dict[predictions.item()]
    
    
    if predicted_label == 'Negatif':
        response = random.choice(positive_responses)
    else:
        response = "Komentar Positif."
    
    return predicted_label, response


def handle_input():
    input_text = entry.get()
    
    if input_text.strip() == "":
        messagebox.showwarning("Input Kosong", "Silakan masukkan komentar!")
        return

    
    predicted_label, response = respond_to_comment(input_text)
    
    
    log_text = f"Komentar: {input_text} | Klasifikasi: {predicted_label} | Balasan: {response}"
    log_listbox.insert(tk.END, log_text)


window = tk.Tk()
window.title("Better Chat System for Mobile Legends")


input_label = tk.Label(window, text="Masukkan komentar:")
input_label.pack(pady=5)
entry = tk.Entry(window, width=50)
entry.pack(pady=5)


submit_button = tk.Button(window, text="Kirim Komentar", command=handle_input)
submit_button.pack(pady=10)


log_listbox = tk.Listbox(window, width=100, height=10)
log_listbox.pack(pady=10)


window.mainloop()
