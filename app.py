from flask import Flask, render_template, request
import numpy as np 
import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn

class BERT_architecture(nn.Module):
    def __init__(self, bert):
        super(BERT_architecture, self).__init__()
        self.bert = bert 
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

app = Flask(__name__,static_folder='static')

# Function to load BERT model and tokenizer
def load_model_and_tokenizer():
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return bert, tokenizer

# Function to preprocess text and get predictions
def get_predictions(sentence, bert, tokenizer):
    # Tokenize input text
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    
    # Get model predictions
    model = BERT_architecture(bert)
    model.load_state_dict(torch.load('saved_weights.pt', map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        preds = model(encoded_input['input_ids'], encoded_input['attention_mask'])
        preds = preds.detach().cpu().numpy()
    
    # Assuming binary classification (0 and 1), convert predictions to readable format
    pred_labels = np.argmax(preds, axis=1)
    result = "Negative" if pred_labels[0] == 0 else "Positive"
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load BERT model and tokenizer
    bert, tokenizer = load_model_and_tokenizer()
    
    # Get user input from form
    sentence = request.form['sentence']
    
    # Get predictions
    prediction = get_predictions(sentence, bert, tokenizer)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
