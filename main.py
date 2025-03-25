import json
import re
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from autocorrect import Speller
import math
import os
import sympy

# Initialize spell checker
spell = Speller(lang='en')

# Ensure necessary NLTK packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

for key, value in data.items():
    if "hostel detail" in value.lower():
        print(f"Found in: {key} → {value}")

# Load and save data functions
def load_data(file_path="data.json"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def save_data(data, file_path="data.json"):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def solve_math_query(query):
    try:
        query = query.replace("^", "**")  # Handle exponents
        result = sympy.sympify(query).evalf()
        return f"The answer is: {result}"
    except Exception as e:
        return f"Sorry, I couldn't solve that math problem. Error: {e}"

# Text cleaning function
def clean_text(text):
    text = text.lower()
    if any(op in text for op in ['+', '-', '*', '/', '^']):  # Keep math expressions unchanged
        return text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = spell(text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# Dataset class for chatbot
class ChatbotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Train the chatbot model
def train_chatbot(data, model_dir="model", epochs=3, batch_size=16, max_len=128):
    texts = list(data.keys())
    labels = list(data.values())
    if not texts or not labels:
        raise ValueError("No training data found!")

    label_map = {label: idx for idx, label in enumerate(set(labels))}
    numeric_labels = [label_map[label] for label in labels]

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, numeric_labels, test_size=0.1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = ChatbotDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = ChatbotDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Save model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(f"{model_dir}/label_map.json", "w") as f:
        json.dump(label_map, f)
    print(f"Model saved to {model_dir}")

    return model, tokenizer, label_map

# Load the trained model
def load_model(model_dir="model"):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    with open(f"{model_dir}/label_map.json", "r") as f:
        label_map = json.load(f)
    return model, tokenizer, label_map

# Handle mathematical queries
def solve_math_query(query):
    try:
        query = query.replace("^", "**")  # Handle exponents
        result = eval(query)
        return f"The answer is: {result}"
    except:
        return "Sorry, I couldn't solve that math problem."

# Generate chatbot response
def chatbot_response(user_input, model, tokenizer, label_map, data):
    user_input = clean_text(user_input)

    # Check if input is a math query
    if any(op in user_input for op in ['+', '-', '*', '/', '^']):
        return solve_math_query(user_input)

    # Check if the input exists in stored data
    if user_input in data:
        response = data[user_input]

        # Filter out "For hostel detail..." responses
        if "hostel detail" in response.lower():
            return "I'm not sure. Can you clarify?"

        return response

    # Use AI model to generate response
    encoding = tokenizer.encode_plus(
        user_input, add_special_tokens=True, max_length=500,
        return_token_type_ids=False, padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='pt'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label_idx = torch.argmax(logits, dim=1).item()

    label_map_rev = {v: k for k, v in label_map.items()}
    return label_map_rev.get(predicted_label_idx, "I'm not sure. Can you clarify?")

# Main chatbot loop
def run_chatbot():
    data = load_data()
    if not data:
        print("No data found. Please add data to 'data.json'.")
        return

    model_dir = "model"
    if not os.path.exists(model_dir):
        print("Training model...")
        model, tokenizer, label_map = train_chatbot(data, model_dir)
    else:
        print("Loading pre-trained model...")
        model, tokenizer, label_map = load_model(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot_response(user_input, model, tokenizer, label_map, data)
        print("Bot:", response)

if __name__ == "__main__":
    run_chatbot()
