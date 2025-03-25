# 🤖 AI GENERATIVE

An AI-powered chatbot implemented in Python using **BERT** for NLP-based responses, advanced text preprocessing, and math query solving. This chatbot learns from user-provided data and improves responses dynamically. 🚀

## ✨ Features
- ✅ **Machine Learning-Powered:** Uses **BERT** for understanding and generating responses.
- 🧹 **Advanced Text Processing:** Includes spell-checking, lemmatization, and stopword removal.
- 🧮 **Math Solver:** Can evaluate mathematical expressions.
- 📚 **Trainable Model:** Learns from user-provided data stored in `data.json`.
- 💾 **Data Storage:** Saves and loads chatbot training data efficiently.
- 🚀 **GPU Acceleration:** Utilizes CUDA if available for faster training.

## 📌 Prerequisites
Ensure you have the following installed:
- 🐍 Python 3.x
- PyTorch
- Transformers (`pip install transformers`)
- NLTK (`pip install nltk`)
- NumPy (`pip install numpy`)
- Scikit-learn (`pip install scikit-learn`)
- Autocorrect (`pip install autocorrect`)

## ▶️ Running the Chatbot
1. 📥 Clone this repository or download the script.
2. 🖥️ Open a terminal and navigate to the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the chatbot:
   ```bash
   python chatbot.py
   ```
5. Chat with the bot! Type `exit` to quit.

## 💡 Example Usage
```
You: hello
Bot: Hey! Need any help?
```
```
You: what is programming
Bot: 🖥️ Algorithms and data structures are key to efficient programming.
```
```
You: 5 + 3 * 2
Bot: The answer is: 11
```
```
You: how are you?
Bot: 🤖 I'm not sure. Can you clarify?
```

## 🛠️ How It Works
1. **Loads Data:** Reads training data from `data.json`.
2. **Cleans Input:** Uses **NLTK** for spell-checking, tokenization, and lemmatization.
3. **Predicts Response:**
   - If input matches stored data, it returns the answer.
   - If input contains a math query, it evaluates the expression.
   - Otherwise, it predicts the closest answer using **BERT**.
4. **Learns & Improves:** The model can be re-trained with new data.

## 🏗️ Code Overview
### 🔹 Data Handling
- Loads data from `data.json`.
- Saves updated knowledge when trained.

### 🔹 Preprocessing
- Converts text to lowercase.
- Removes special characters and stopwords.
- Uses a spell checker and lemmatizer.

### 🔹 Model Training
- Uses **BERT** for training with labeled chatbot data.
- Optimized using **AdamW** optimizer.
- Saves model and tokenizer after training.

### 🔹 Chatbot Response
- Checks for predefined responses first.
- Uses **BERT** to predict the closest response.
- Handles mathematical expressions dynamically.

### 📝 Code Snippet
```python
import random
from transformers import BertTokenizer, BertForSequenceClassification

def chatbot_response(user_input, model, tokenizer, label_map, data):
    user_input = clean_text(user_input)
    if user_input in data:
        return data[user_input]
    encoding = tokenizer.encode_plus(user_input, return_tensors='pt', padding='max_length', truncation=True)
    outputs = model(**encoding)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return label_map.get(predicted_label, "I'm not sure. Can you clarify?")
```

## 🔮 Future Improvements
- 🤖 Implement **self-learning** by updating responses automatically.
- 🌐 Integrate **web-based interface** for a more interactive chatbot experience.
- 🏗️ Enhance **NLP capabilities** for better contextual understanding.

## 📜 License
This project is open-source and free to use under the MIT License.


To add this to your Hugging Face model page, you can follow these steps:

1. Go to the Hugging Face model page for your chatbot: [AI GENERATIVE](https://huggingface.co/TUHIN-PROGRAMMER/AI_GENARATIVE).
2. Click on the "Edit" button.
3. Copy and paste the above `README.md` into the description field.
4. Save your changes.

This will ensure your Hugging Face page is well-documented and ready for others to use! Let me know if you'd like further modifications or additions.


