# 🍿 Movie Review Analyzer – Sentiment Analysis with BiLSTM

**Movie Review Analyzer** is a deep learning-based Streamlit web app that performs **sentiment analysis** on user-submitted movie reviews. Powered by a **Bidirectional LSTM neural network**, this model can understand the emotional tone of reviews and classify them as either **positive** or **negative**.

> _"The movie was a complete masterpiece!"_ → ✅ **Positive**  
> _"Worst movie I’ve ever seen."_ → ❌ **Negative**

---

## 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/🚀%20Launch%20App-Popcorn%20Pulse-red?style=flat-square&logo=streamlit)](https://popcornpulse.streamlit.app/)

> Try it live and test with your own movie reviews!

---

## 🧠 How It Works

The sentiment classification model uses the following deep learning architecture:

```python
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_seq_len - 1))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(1, activation='sigmoid'))
```

### 🔍 Layer-wise Breakdown

- 🔤 **Tokenizer**: Converts text into sequences of word indices.
- 🧠 **Embedding Layer**: Learns dense vector representations of words.
- 🔁 **Bidirectional LSTM**: Captures both past and future context in text.
- 🎯 **Dense Sigmoid Output**: Predicts a binary sentiment score (positive/negative).

---

### 💡 Features

- ✅ Classifies movie reviews into **positive** or **negative**
- 🧠 Built using deep learning with **Bidirectional LSTM**
- 🔤 Tokenizer-based text preprocessing
- 📦 Easy-to-use **Streamlit** web interface
- 📱 Fully responsive and interactive
- 🌐 Deployed live at [popcornpulse.streamlit.app](https://popcornpulse.streamlit.app/)

---

### 🛠️ Tech Stack

| Tool              | Purpose                                |
|-------------------|----------------------------------------|
| Python            | Core programming language              |
| TensorFlow / Keras| Deep learning model framework          |
| NLTK              | Natural Language Toolkit for cleaning  |
| Streamlit         | Web app frontend                       |
| NumPy & Pandas    | Data manipulation and preprocessing    |



```bash
git clone https://github.com/Jatin-GI/movie-review-analyzer.git
cd movie-review-analyzer

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
## 📄 License

[![MIT License](https://img.shields.io/github/license/Jatin-GI/movie-review-analyzer?style=flat-square)](LICENSE)

This project is licensed under the **MIT License**.  
You are free to **use**, **modify**, and **distribute** this project for personal or commercial purposes.  
See the [LICENSE](LICENSE) file for more details.

---

## 👤 Author

**Developed with 💻 & ❤️ by [Jatin Gupta](https://github.com/Jatin-GI)**

[![GitHub followers](https://img.shields.io/github/followers/Jatin-GI?style=social)](https://github.com/Jatin-GI)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/jatin-gupta-b02b37292)
[![Email](https://img.shields.io/badge/Email-guptajatin0416@gmail.com-red?style=flat-square&logo=gmail)](mailto:guptajatin0416@gmail.com)

📫 Feel free to reach out, collaborate, or just say hi!

---

## 🙌 Support & Contributions

If you like this project, please consider:

[![GitHub Stars](https://img.shields.io/github/stars/Jatin-GI/movie-review-analyzer?style=social)](https://github.com/Jatin-GI/movie-review-analyzer/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Jatin-GI/movie-review-analyzer?style=social)](https://github.com/Jatin-GI/movie-review-analyzer/forks)

Pull requests, suggestions, and feedback are always welcome!  
Let’s connect, collaborate, and build something amazing together. ✨

---

## 📢 Stay Tuned

[![GitHub last commit](https://img.shields.io/github/last-commit/Jatin-GI/movie-review-analyzer?style=flat-square)](https://github.com/Jatin-GI/movie-review-analyzer/commits)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)

🚀 More projects like this are coming soon.  
Follow me on [GitHub](https://github.com/Jatin-GI) or [LinkedIn](https://www.linkedin.com/in/jatin-gupta-b02b37292) to stay updated!
