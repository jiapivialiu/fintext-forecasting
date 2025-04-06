from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

bigdata_train = load_dataset("TheFinAI/flare-sm-bigdata", split="train")
acl_train = load_dataset("TheFinAI/flare-sm-acl", split="train")
cikm_train = load_dataset("TheFinAI/flare-sm-cikm", split="train")

bigdata_valid = load_dataset("TheFinAI/flare-sm-bigdata", split="validation")
acl_valid = load_dataset("TheFinAI/flare-sm-acl", split="valid")
cikm_valid = load_dataset("TheFinAI/flare-sm-cikm", split="valid")

bigdata_test = load_dataset("TheFinAI/flare-sm-bigdata", split="test")
acl_test = load_dataset("TheFinAI/flare-sm-acl", split="test")
cikm_test = load_dataset("TheFinAI/flare-sm-cikm", split="test")

bigdata_train_df = bigdata_train.to_pandas()[['gold', 'text']] # 0: rise, 1: fall
acl_train_df = acl_train.to_pandas()[['gold', 'text']] 
cikm_train_df = cikm_train.to_pandas()[['gold', 'text']]

bigdata_valid_df = bigdata_valid.to_pandas()[['gold', 'text']]
acl_valid_df = acl_valid.to_pandas()[['gold', 'text']]
cikm_valid_df = cikm_valid.to_pandas()[['gold', 'text']]

bigdata_test_df = bigdata_test.to_pandas()[['gold', 'text']]
acl_test_df = acl_test.to_pandas()[['gold', 'text']]
cikm_test_df = cikm_test.to_pandas()[['gold', 'text']]

# fine tune financial LLM
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained financial model and tokenizer
model_name = 'ProsusAI/finbert'  # Updated to a better financial pre-trained LLM
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize the model
finbert = BertForSequenceClassification.from_pretrained(model_name)

# Update the model configuration for binary classification
finbert.config.num_labels = 2  # Set the number of labels to 2 for binary classification

# Replace the classifier layer to match the updated number of labels
finbert.classifier = nn.Linear(finbert.config.hidden_size, finbert.config.num_labels)

# Ensure the model's forward pass uses the updated classifier
finbert.num_labels = 2  # Explicitly set the number of labels in the model

# Print model configuration for debugging
print("Updated model configuration:")
print(f"Number of labels: {finbert.config.num_labels}")
print(f"Classifier output shape: {finbert.classifier.out_features if hasattr(finbert.classifier, 'out_features') else 'Custom classifier'}")

# Configure PEFT with LoRA specifically for binary classification
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                       # Reduced rank for efficiency
    lora_alpha=16,             # Adjusted alpha for better stability
    lora_dropout=0.1,
    bias="none",               # No bias adaptation for classification tasks
    target_modules=["query", "key", "value", "output.dense"],  # Target more modules for better adaptation
    modules_to_save=["classifier"]  # Save the classifier layer which is crucial for the task
)

# Apply PEFT to the model
finbert = get_peft_model(finbert, peft_config)
finbert.print_trainable_parameters()

# Define a custom dataset to handle the text and labels
class FinancialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):  # Reduced max_len for efficiency
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])  # Ensure text is a string
        label = int(self.labels[item])  # Ensure label is an integer
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
# Sample data (financial texts with labels for rise or fall)
train_texts = bigdata_train_df['text'].tolist()
train_labels = bigdata_train_df['gold'].tolist()  # 1: Rise, 0: Fall
valid_texts = bigdata_valid_df['text'].tolist()
valid_labels = bigdata_valid_df['gold'].tolist()

print(f"Number of training examples: {len(train_texts)}")
print(f"Number of validation examples: {len(valid_texts)}")
print(f"Class distribution in training data: {np.bincount(train_labels)}")
print(f"Class distribution in validation data: {np.bincount(valid_labels)}")

# Prepare the dataset and dataloaders
train_dataset = FinancialDataset(train_texts, train_labels, tokenizer)
val_dataset = FinancialDataset(valid_texts, valid_labels, tokenizer)

# Use a larger batch size if your hardware allows
batch_size = 8  # Increased for faster training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the optimizer and scheduler for better convergence
optimizer = AdamW(finbert.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)

# Add a learning rate scheduler
total_steps = len(train_loader) * 5  # 5 epochs
warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Move the device definition earlier in the code, before the loss function definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert = finbert.to(device)
print(f"Using device: {device}")

# Loss function with class weights if needed (if dataset is imbalanced)
if np.bincount(train_labels)[0] != np.bincount(train_labels)[1]:
    # Calculate class weights for balanced loss
    class_counts = np.bincount(train_labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using weighted loss with weights: {class_weights}")
else:
    loss_fn = nn.CrossEntropyLoss()
    print("Using standard unweighted loss")

# Fix the train function to handle potential errors and match the arguments
def train(model, train_loader, optimizer, scheduler, device, epoch):
    """
    Training function for one epoch.
    """
    print(f"\nStarting training epoch {epoch+1}...")
    print(f"Total batches: {len(train_loader)}")
    
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % 50 == 0:
            local_tz = pytz.timezone('America/Vancouver')
            local_time = datetime.now(local_tz)
            print(f"Processing batch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {epoch_loss/(batch_idx+1) if batch_idx > 0 else 0:.4f}, "
                  f"Accuracy: {correct_predictions/total_predictions*100 if total_predictions > 0 else 0:.2f}%, "
                  f"Time: {local_time.strftime('%H:%M:%S')}")

        try:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Print shapes for debugging on first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
            
            # Forward pass - ensure we're passing the correct inputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Print shapes for debugging on first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"Output shapes - logits: {logits.shape}")
                print(f"Unique labels: {torch.unique(labels)}")  # Verify we have binary labels
            
            # Calculate batch accuracy
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            # Print more debugging info
            print(f"Input shapes: input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
            print(f"Label values: {labels}")
            raise  # Re-raise the exception after printing debug info
    
    # Calculate epoch metrics
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, accuracy

# Evaluation loop with detailed metrics
def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate overall metrics
    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(true_labels, predictions, decimal=10)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'loss': avg_val_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Plotting function to visualize training progress
def plot_training_progress(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, epoch):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')
    plt.close()

# Fine-tune and evaluate
epochs = 5
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_f1 = 0
best_model_path = "best_finbert_model.pt"

# Apply original FinBERT on validation set for baseline comparison
print("\nEvaluating original FinBERT model on validation set...")
original_finbert = BertForSequenceClassification.from_pretrained(model_name)
original_finbert.config.num_labels = 2
original_finbert.classifier = nn.Linear(original_finbert.config.hidden_size, 2)
original_finbert = original_finbert.to(device)

original_metrics = evaluate(original_finbert, val_loader, device)
print("\nOriginal FinBERT Performance on Validation Set:")
print(f"Accuracy: {original_metrics['accuracy']:.4f}")
print(f"Precision: {original_metrics['precision']:.4f}")
print(f"Recall: {original_metrics['recall']:.4f}")
print(f"F1 Score: {original_metrics['f1']:.4f}")
print(f"Confusion Matrix:\n{original_metrics['confusion_matrix']}")

# Try a single batch first to debug
print("Testing a single batch through the model for debugging...")
for batch in train_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    print(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
    print(f"Unique labels: {torch.unique(labels)}")
    
    outputs = finbert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"Output shapes - logits: {outputs.logits.shape}")
    
    # Check if the number of classes matches our expectations
    if outputs.logits.shape[1] != 2:
        print(f"WARNING: Model is outputting {outputs.logits.shape[1]} classes, but we need 2 for binary classification.")
    
    break  # Just test one batch

for epoch in range(epochs):
    # Training phase - use the updated train function with all required arguments
    train_loss, train_acc = train(
        model=finbert,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=epoch
    )
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluation phase
    val_metrics = evaluate(finbert, val_loader, device)
    val_losses.append(val_metrics['loss'])
    val_accs.append(val_metrics['accuracy'])
    
    # Plot confusion matrix
    plot_confusion_matrix(val_metrics['confusion_matrix'], epoch)
    
    # Print metrics
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.10f}, Train Accuracy: {train_acc:.10f}")
    print(f"Val Loss: {val_metrics['loss']:.10f}, Val Accuracy: {val_metrics['accuracy']:.10f}")
    
    # Save the best model based on F1 score
    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        # Save the entire model, including PEFT configuration
        torch.save({
            'model_state_dict': finbert.state_dict(),
            'peft_config': peft_config
        }, best_model_path)
        print(f"New best model saved with F1 score: {best_f1:.4f}")

# Plot training progress
plot_training_progress(train_losses, train_accs, val_losses, val_accs)

# Load the best model for final evaluation
checkpoint = torch.load(best_model_path)
finbert.load_state_dict(checkpoint['model_state_dict'])  # Load model state
peft_config = checkpoint['peft_config']  # Reload PEFT configuration if needed
finbert.eval()

# Evaluate on test set
test_texts = bigdata_test_df['text'].tolist()
test_labels = bigdata_test_df['gold'].tolist()
test_dataset = FinancialDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

test_metrics = evaluate(finbert, test_loader, device)
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1 Score: {test_metrics['f1']:.4f}")
print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

# Generate and print classification report
test_predictions = []
test_true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = finbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=1)
        test_predictions.extend(preds.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(test_true_labels, test_predictions, target_names=['Fall', 'Rise']))

# Save the final model
finbert.save_pretrained("finbert_binary_classifier")
tokenizer.save_pretrained("finbert_binary_classifier")
print("\nFinal model saved to 'finbert_binary_classifier'")