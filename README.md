# 🌿 ArgiVeda

ArgiVeda is an intelligent, multimodal plant disease diagnosis and advisory system built for low-end Android devices. It combines high-accuracy Convolutional Neural Networks (CNNs) and Retrieval-Augmented Generation (RAG) to provide real-time, grounded agricultural advice tailored to Indian farmers.

---

## 🚀 Features

- 🌱 **Plant Disease Detection**
  - Custom CNN model trained on the PlantVillage dataset.
  - 97.7% test accuracy across 38 disease classes.
  - Lightweight ONNX model (~29.9MB) supports offline classification.

- 🧠 **Contextual AI Advisory**
  - Retrieval-Augmented Generation (RAG) powered by OpenAI + Pinecone.
  - Cites trusted agronomy KB sources: pest guides, treatment manuals, etc.
  - Enables follow-up questions and local context understanding.

- 📱 **Farmer-Friendly Interface**
  - Supports 13 Indian languages and voice input.
  - Intuitive mobile UI designed for low literacy and low-bandwidth usage.
  - Designed for sub-5s latency on budget Android devices.

- 🌐 **Offline Support**
  - Offline image classification using ONNX.
  - Caches top-K retrieval responses for offline Q&A.

- 🧩 **Modular, Scalable Architecture**
  - Microservices: `image-api`, `rag-service`, `chat-ui`.
  - Easily extendable with new crops, diseases, or knowledge documents.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, ONNX Runtime
- **AI Models**: Custom CNN, OpenAI GPT
- **RAG**: Pinecone Vector DB + OpenAI
- **Frontend**: TypeScript / React Native

---

## 🧪 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/argiveda.git
cd argiveda
````

### 2. Install Python dependencies

### 3. Set environment variables

See `.env.example` below or create your own `.env`.

### 4. Run backend server

```bash
python app.py
```
### 5. Run index.html

---

## 🔐 Environment Variables

Create a `.env` file in your root directory with the following contents:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_env
PINECONE_INDEX_NAME=your_index_name

# App Settings
PORT=8000
DEBUG=True
```

---

## 📸 Demo

https://github.com/user-attachments/assets/ed18a0c1-f2c5-4dd8-80b7-2b0e9d8db4d0

---

## 📄 References

* [PlantVillage Dataset](https://plantvillage.psu.edu/)
* [Lewis et al. (2020) - RAG for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401)
* [Srivastava et al. (2014) - Dropout](https://jmlr.csail.mit.edu/papers/volume15/srivastava14a/srivastava14a.pdf)
* [Ioffe & Szegedy (2015) - Batch Normalization](https://arxiv.org/abs/1502.03167)
* [TNAU Agritech Portal](http://agritech.tnau.ac.in)

---


## 🪴 License

This project is licensed under the **MIT License**.

---
```
