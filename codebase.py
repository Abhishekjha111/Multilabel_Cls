
# Import necessary libraries
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#Here, we import libraries and modules needed for our text classification task, including the Hugging Face Transformers library (for BERT model and tokenizer), PyTorch, pandas for data manipulation, numpy for numerical operations, and scikit-learn for metrics like accuracy and F1 score.

# Load the dataset
data = pd.read_csv("PubMed Multi Label Text Classification Dataset Processed.csv")


#We load our dataset from a CSV file named "PubMed Multi Label Text Classification Dataset Processed.csv" using pandas and store it in the 'data' DataFrame.

# Combine text data (Title and abstractText)
data['text_data'] = data['Title'] + " " + data['abstractText']

#We create a new column in the DataFrame called 'text_data' by combining the 'Title' and 'abstractText' columns. This new column contains the text data we'll use for classification.

# Extract labels
labels = data.iloc[:, 6:]

#We extract the labels for our classification task. 'labels' now contains the target labels for each text.

# Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=labels.shape[1])
model = BertForSequenceClassification.from_pretrained(model_name, config=config)


#We load the BioBERT model and tokenizer. BioBERT is a specialized BERT model for biomedical and clinical text. We create a tokenizer and configuration for the model, and then we initialize our classification model using this configuration.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text_data'], labels, test_size=0.2, random_state=42)


#We split our data into training and testing sets using the `train_test_split` function from scikit-learn. 'X_train' and 'y_train' contain the training data and labels, while 'X_test' and 'y_test' contain the testing data and labels.

max_length = 128

encodings_train = tokenizer(X_train.tolist(), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', is_split_into_words=True)
encodings_test = tokenizer(X_test.tolist(), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', is_split_into_words=True)

#We tokenize and encode the training and testing text data using the BERT tokenizer. We set the maximum sequence length to 128, pad sequences to the same length, and ensure the output is in PyTorch tensors.
# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()


#We define an optimizer for updating the model's parameters and a loss function for measuring the error between the model's predictions and the true labels. We use the BCEWithLogitsLoss, which is commonly used for multilabel classification tasks.
# Training loop
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    output = model(**encodings_train)
    loss = loss_fn(output.logits, y_train.float())
    loss.backward()
    optimizer.step()
#We start the training loop. For each epoch (complete pass through the training data), we set the model to training mode and calculate predictions ('output') for the training data. We compute the loss and perform backpropagation to update the model's parameters using the optimizer.
# Evaluation
model.eval()
with torch.no_grad():
    test_output = model(**encodings_test)
    predicted_labels = torch.sigmoid(test_output.logits)


#We switch the model to evaluation mode and make predictions on the testing data. We use torch.sigmoid to convert the model's logits to probabilities. We don't update the model's parameters during evaluation (hence the `torch.no_grad()` context).

# Calculate accuracy and F1 score for each label
accuracies = []
f1_scores = []
for i in range(y_test.shape[1]):
    accuracy_i = accuracy_score(y_test.iloc[:, i], (predicted_labels[:, i] > 0.5))
    f1_i = f1_score(y_test.iloc[:, i], (predicted_labels[:, i] > 0.5), average='binary')
    accuracies.append(accuracy_i)
    f1_scores.append(f1_i)

#We calculate accuracy and F1 score for each label in a multilabel classification fashion. The loop goes through each label, comparing predicted values to the true labels.

# Calculate micro-averaged F1 score
f1_micro = f1_score(y_test, (predicted_labels > 0.5), average='micro')


#We calculate the micro-averaged F1 score, which considers all labels together, providing a single F1 score for the overall classification performance.

# Print evaluation metrics
print("Accuracy (Micro-Averaged): {:.2f}%".format(accuracy * 100))
print("F1 Score (Micro): {:.2f}".format(f1_micro))
print("Individual Label Metrics:")
for label, accuracy_i, f1_i in zip(labels.columns, accuracies, f1_scores):
    print(f"Label: {label}, Accuracy: {accuracy_i:.2f}%, F1 Score: {f1_i:.2f}")

#We print out the evaluation metrics, including the micro-averaged accuracy, micro-averaged F1 score, and individual label metrics (accuracy and F1 score for each label). The results help assess how well the model is performing in classifying the medical texts.
