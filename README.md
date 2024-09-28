
# TinyBERT Enhanced Chat System for Mobile Legends

I developed a TinyBERT-powered chat moderation system for Mobile Legends. It classifies in-game comments as positive, neutral, or negative, providing real-time feedback to curb toxic behavior with dynamic responses and efficient sentiment analysis, enhancing the gaming experience

This project is still under development and is expected to be completed more quickly. The approach taken will focus more on improving the dataset to produce better classification.
## 🎬 Demo

![Demo](https://github.com/user-attachments/assets/4c92c786-2bb0-44bc-a636-264508bf5aab)


## 🔨 Installation

PyPi version

![IPyKernel version](https://img.shields.io/badge/IPyKernel-v6.29.4-yellow)
![Pandas version](https://img.shields.io/badge/pandas-v2.2.3-black)
![Pytorch version](https://img.shields.io/badge/pytorch-v2.4.1+cu118-red) 
![ScikitLearn version](https://img.shields.io/badge/scikitlearn-v1.5.2-blue)

First make sure you have install PyTorch with CUDA support, because it will take forever for training BERT model. 
```bash
  import torch
  print(torch.cuda.is_available())
```
If the output is false, then you need to check your driver version before implementing CUDA. example for installing CUDA 11.8
```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
First create venv for this folder, you can just copy and paste this :
```bash
  python --version
  python -m venv nama_venv
  nama_venv\Scripts\activate
  pip install pandas
  pip install scikit-learn
  pip install transformers
  pip install seaborn
```

I have a problem regarding the time spent on TinyBERT training. with an RTX 2060 Max Q Design GPU it takes around 73 minutes even with CUDA support. 

Actually, my plan is to use IndoBERT to make this project. However, due to performance limitations, we had to change plans and use TinyBERT. This is the setup if u want to use IndoBERT

```bash
from transformers import BertForSequenceClassification, Trainer, TrainingArguments


model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p2", num_labels=3)

memaksa penggunaan CPU
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=64, 
    warmup_steps=500, 
    weight_decay=0.01, 
    logging_dir='./logs', 
    logging_steps=10, 
    evaluation_strategy="epoch", 
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()
```

but in the end I used TinyBERT, so I modified the code to load pre-trained TinyBERT for classification, like this :

```bash
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=128)


train_dataset = SentimentDataset(train_encodings, train_labels.tolist())
test_dataset = SentimentDataset(test_encodings, test_labels.tolist())


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()

```

this is the performance of my laptop during model training.


![{info}](https://github.com/user-attachments/assets/2488e079-6d0f-411c-a40d-212b233750f3)


## Result

The results I have obtained are quite satisfying with the following outcomes (note that this project is still in development) :

![Accuracy](https://img.shields.io/badge/Accuracy-0.90118-black)

![Precision](https://img.shields.io/badge/Precision-0.8832453423991299-black)

![Recall](https://img.shields.io/badge/Recall-0.90118-black) 

![F1-score](https://img.shields.io/badge/F1score-0.8861208356918333-black)


## Features
- Accurate classification results
- A strict system that enforces bans when offensive language is detected


## Authors

<img alt="Static Badge" src="https://img.shields.io/badge/AgungHari-black?style=social&logo=github&link=https%3A%2F%2Fgithub.com%2FAgungHari">

## License

<img alt="GitHub License" src="https://img.shields.io/github/license/AgungHari/TinyBERT-Enhanced-Chat-System-for-Mobile-Legends">


