# Neural nets on embedded text with JAX
from datasets import load_dataset

#* Data Loading *#

acl_train = load_dataset("TheFinAI/flare-sm-acl", split="train")
acl_valid = load_dataset("TheFinAI/flare-sm-acl", split="valid")
acl_test = load_dataset("TheFinAI/flare-sm-acl", split="test")

# * Data Preprocessing *#

# Select columns
acl_train_df = acl_train.to_pandas()[['gold', 'text']] # 0: rise, 1: fall
acl_valid_df = acl_valid.to_pandas()[['gold', 'text']]
acl_test_df = acl_test.to_pandas()[['gold', 'text']]

# Initialize the pre-trained model for text embedding (using a compact and efficient model)
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text embedding with batch processing
def batch_encode(texts, batch_size=32):
    def get_sbert_embeddings(texts):
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = get_sbert_embeddings(batch)
        embeddings_list.append(batch_embeddings)
    return np.vstack(embeddings_list)

import jax
import jax.numpy as jnp

# training data processing
texts = acl_train_df['text'].tolist()
X_train_embeddings = batch_encode(texts)
X_train_embeddings = np.array(X_train_embeddings, dtype=np.float32)
acl_train_embedded = np.concatenate([acl_train_df[['gold']], X_train_embeddings], axis=1)
acl_train_embedded_jax = jnp.array(acl_train_embedded, dtype=jnp.float32)

# validation data processing
valid_texts = acl_valid_df['text'].tolist()
X_valid_embeddings = batch_encode(valid_texts)
X_valid_embeddings = np.array(X_valid_embeddings, dtype=np.float32)
acl_valid_embedded = np.concatenate([acl_valid_df[['gold']], X_valid_embeddings], axis=1)
acl_valid_embedded_jax = jnp.array(acl_valid_embedded, dtype=jnp.float32)

# test data processing
test_texts = acl_test_df['text'].tolist()
X_test_embeddings = batch_encode(test_texts)
X_test_embeddings = np.array(X_test_embeddings, dtype=np.float32)
acl_test_embedded = np.concatenate([acl_test_df[['gold']], X_test_embeddings], axis=1)
acl_test_embedded_jax = jnp.array(acl_test_embedded, dtype=jnp.float32)

# * Modelling *#

#* 1. XGBoost (Baseline Model) *#
from sklearn.metrics import classification_report, matthews_corrcoef
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np

# Prepare data for modelling
y_train = acl_train_embedded[:, 0].reshape(-1, 1)
X_train = acl_train_embedded[:, 1:]
y_valid = acl_valid_embedded[:, 0].reshape(-1, 1)
X_valid = acl_valid_embedded[:, 1:]

# Check class balance
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("\nClass distribution in training data:", class_distribution)

# Apply SMOTE for class balancing if imbalanced
if len(unique) > 1 and min(counts) / max(counts) < 0.8:
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train.ravel())
    # Convert back to original shape
    X_train = X_train_resampled
    y_train = y_train_resampled.reshape(-1, 1)
    print("After SMOTE - samples:", X_train.shape[0])

# Create a pipeline with standardization and XGBoost
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Define parameter grid for tuning
param_grid = {
    'clf__n_estimators': [50, 100, 200, 300],
    'clf__max_depth': [3, 4, 5, 6, 8],
    'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'clf__subsample': [0.6, 0.8, 1.0],
    'clf__colsample_bytree': [0.6, 0.8, 1.0],
    'clf__min_child_weight': [1, 3, 5],
    'clf__gamma': [0, 0.1, 0.2]
}

# Use RandomizedSearchCV for efficient hyperparameter tuning
print("\nTuning XGBoost hyperparameters...")
random_search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=20,  # Number of parameter settings to try
    cv=5,
    scoring='matthews_corrcoef',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train.ravel())

# Get best parameters and model
print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best cross-validation MCC: {random_search.best_score_:.4f}")

# Train optimized model with best parameters
best_xgb_model = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_xgb_model.predict(X_valid)

# Calculate MCC for XGBoost
xgb_mcc = matthews_corrcoef(y_valid, y_pred)

# Print performance metrics
print("\nOptimized XGBoost Results:")
print(f"Validation Accuracy: {best_xgb_model.score(X_valid, y_valid):.4f}")
print(f"Matthews Correlation Coefficient: {xgb_mcc:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_valid, y_pred))

# Feature importance analysis
if hasattr(best_xgb_model[-1], 'feature_importances_'):
    importances = best_xgb_model[-1].feature_importances_
    print("\nTop 10 Most Important Features:")
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Store the model for comparison with other methods
baseline_model = best_xgb_model

#* 2. MLP with JAX *#
import jax
import jax.numpy as jnp
from jax import random
from typing import List, Tuple, Any
import optax  # For Adam optimizer
from functools import partial

# Prepare data for modelling
y_train = acl_train_embedded_jax[:, 0].reshape(-1, 1)
X_train = acl_train_embedded_jax[:, 1:]
y_valid = acl_valid_embedded_jax[:, 0].reshape(-1, 1)
X_valid = acl_valid_embedded_jax[:, 1:]
input_dim = X_train.shape[1]  # Get input dimension from training data

# Define MLP in JAX
def init_mlp_params(layer_sizes: List[int], key: Any) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    params = []
    keys = random.split(key, len(layer_sizes))
    for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w_key, b_key = random.split(k)
        W = random.normal(w_key, (in_dim, out_dim)) * jnp.sqrt(2. / in_dim)
        b = jnp.zeros((out_dim,))
        params.append((W, b))
    return params

# Enhanced MLP with dropout
def mlp_forward(params: List[Tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray, dropout_rate: float = 0.0, 
               train: bool = False, key: Any = None) -> jnp.ndarray:
    """Forward pass with dropout support"""
    for i, (W, b) in enumerate(params[:-1]):
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
        
        # Apply dropout during training
        if train and dropout_rate > 0:
            if key is None:
                raise ValueError("Random key required for dropout")
            dropout_key = random.fold_in(key, i)  # Different key for each layer
            mask = random.bernoulli(dropout_key, p=1-dropout_rate, shape=x.shape)
            x = x * mask / (1 - dropout_rate)  # Scale to maintain expected value
            
    W_last, b_last = params[-1]
    logits = jnp.dot(x, W_last) + b_last
    return logits

# Loss and training step
def binary_cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    preds = jax.nn.sigmoid(logits)
    return -jnp.mean(labels * jnp.log(preds + 1e-7) + (1 - labels) * jnp.log(1 - preds + 1e-7))

# Improved training step with Adam optimizer
@partial(jax.jit, static_argnums=(4, 5))
def train_step(params, X_batch, y_batch, opt_state, dropout_rate=0.2, train=True):
    """Single training step with Adam optimizer and dropout"""
    key = random.PRNGKey(0)  # For reproducibility
    
    def loss_fn(p):
        logits = mlp_forward(p, X_batch, dropout_rate=dropout_rate, train=train, key=key)
        return binary_cross_entropy_loss(logits, y_batch)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# Modified evaluation function for hyperparameter tuning
def evaluate(params, X, y, dropout_rate=0.0, train=False) -> tuple:
    """Evaluate model accuracy and MCC with optional dropout"""
    key = random.PRNGKey(99) if train else None
    logits = mlp_forward(params, X, dropout_rate=dropout_rate, train=train, key=key)
    preds = jax.nn.sigmoid(logits)
    binary_preds = (preds > 0.5).astype(jnp.float32)
    accuracy = jnp.mean(binary_preds == y)
    
    # Calculate MCC
    # Need to convert to numpy arrays for matthews_corrcoef
    y_np = np.array(y.flatten())
    preds_np = np.array(binary_preds.flatten())
    mcc = matthews_corrcoef(y_np, preds_np)
    
    return float(accuracy), float(mcc)

# Hyperparameter tuning
def tune_hyperparameters():
    best_accuracy = 0.0
    best_mcc = -1.0  # MCC ranges from -1 to 1, so start with worst possible
    best_params = None
    best_config = {}
    
    # Define hyperparameter search space
    learning_rates = [1e-4, 5e-4, 1e-3]
    hidden_layer_configs = [
        [128, 64],
        [256, 128],
        [128, 64, 32]
    ]
    dropout_rates = [0.0, 0.2, 0.3, .4, .5, .6, .7, .8, .9]
    batch_sizes = [32, 64, 128]  # For mini-batch training
    
    results = []
    
    for lr in learning_rates:
        for hidden_layers in hidden_layer_configs:
            for dropout_rate in dropout_rates:
                for batch_size in batch_sizes:
                    print(f"\nTrying: lr={lr}, layers={hidden_layers}, dropout={dropout_rate}, batch_size={batch_size}")
                    
                    # Initialize model
                    key = random.PRNGKey(42)
                    layer_sizes = [input_dim] + hidden_layers + [1]
                    params = init_mlp_params(layer_sizes, key)
                    
                    # Initialize optimizer
                    global optimizer  # Make it accessible in train_step
                    optimizer = optax.adam(learning_rate=lr)
                    opt_state = optimizer.init(params)
                    
                    # Mini-batch training
                    num_batches = max(1, len(X_train) // batch_size)
                    
                    for epoch in range(50):  # Fewer epochs for tuning
                        # Shuffle data
                        perm = random.permutation(key, len(X_train))
                        key = random.fold_in(key, epoch)  # Update key for next epoch
                        
                        # Mini-batch updates
                        total_loss = 0.0
                        for i in range(num_batches):
                            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                            X_batch = X_train[batch_idx]
                            y_batch = y_train[batch_idx]
                            
                            params, opt_state, loss = train_step(
                                params, X_batch, y_batch, opt_state, 
                                dropout_rate=dropout_rate, train=True
                            )
                            total_loss += loss
                        
                        avg_loss = total_loss / num_batches
                        if epoch % 10 == 0:
                            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
                    
                    # Evaluate on validation set
                    val_accuracy, val_mcc = evaluate(params, X_valid, y_valid, dropout_rate=0.0, train=False)
                    print(f"Validation Accuracy: {val_accuracy:.4f}, MCC: {val_mcc:.4f}")
                    
                    # Record result
                    config = {
                        'learning_rate': lr,
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'batch_size': batch_size,
                        'val_accuracy': val_accuracy,
                        'val_mcc': val_mcc
                    }
                    results.append(config)
                    
                    # Update best model - now using MCC as primary metric
                    if val_mcc > best_mcc:
                        best_mcc = val_mcc
                        best_accuracy = val_accuracy
                        best_params = params
                        best_config = config
    
    print("\n=== Hyperparameter Tuning Results ===")
    for i, res in enumerate(sorted(results, key=lambda x: x['val_mcc'], reverse=True)):
        print(f"{i+1}. MCC: {res['val_mcc']:.4f}, Acc: {res['val_accuracy']:.4f} - LR: {res['learning_rate']}, " 
              f"Layers: {res['hidden_layers']}, Dropout: {res['dropout_rate']}, "
              f"Batch Size: {res['batch_size']}")
    
    print(f"\nBest Configuration: {best_config}")
    return best_params, best_config

# Run hyperparameter tuning
print("\n=== Starting Hyperparameter Tuning ===")
best_params, best_config = tune_hyperparameters()

# Train final model with best hyperparameters
print("\n=== Training Final Model with Best Hyperparameters ===")
key = random.PRNGKey(0)
layer_sizes = [input_dim] + best_config['hidden_layers'] + [1]
params = init_mlp_params(layer_sizes, key)

# Initialize optimizer with best learning rate
optimizer = optax.adam(learning_rate=best_config['learning_rate'])
opt_state = optimizer.init(params)

# Training with mini-batches
num_epochs = 100
batch_size = best_config['batch_size']
num_batches = max(1, len(X_train) // batch_size)

# Add early stopping parameters
best_val_mcc = -1.0
patience = 5
early_stop_counter = 0
best_params_state = None

# Set seed for reproducibility in training
seed = 258
key = random.PRNGKey(seed)
for epoch in range(num_epochs):
    # Shuffle data
    perm = random.permutation(key, len(X_train))
    key = random.fold_in(key, epoch)
    
    # Mini-batch updates
    total_loss = 0.0
    for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        params, opt_state, loss = train_step(
            params, X_batch, y_batch, opt_state, 
            dropout_rate=best_config['dropout_rate'], train=True
        )
        total_loss += loss
    
    avg_loss = total_loss / num_batches
    
    # Evaluate model on training and validation sets
    train_acc, train_mcc = evaluate(params, X_train, y_train, dropout_rate=0.0, train=False)
    val_acc, val_mcc = evaluate(params, X_valid, y_valid, dropout_rate=0.0, train=False)
    
    # Print progress every 10 epochs or on the last epoch
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Train Mcc: {train_mcc:.4f} | Val Acc: {val_acc:.4f} | Val MCC: {val_mcc:.4f}")
    
    # Early stopping check based on MCC
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        early_stop_counter = 0
        # Save best parameters (deepcopy equivalent for JAX)
        best_params_state = [(W.copy(), b.copy()) for W, b in params]
        print(f"Epoch {epoch+1} | New best MCC: {best_val_mcc:.4f}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Restore best parameters
if best_params_state is not None:
    params = best_params_state
    print(f"Restored best model with validation MCC: {best_val_mcc:.4f}")

# Final evaluation
final_accuracy, final_mcc = evaluate(params, X_valid, y_valid, dropout_rate=0.0, train=False)
print(f"\nFinal Validation Accuracy: {final_accuracy:.10f}")
print(f"Final Validation MCC: {final_mcc:.10f}")

#* 3. Bagging *#

# manually bagging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import numpy as np

# List of classifiers to use
classifiers = [
    LogisticRegression(),
    SVC(kernel='linear'),
    RandomForestClassifier(n_estimators=100),
    KNeighborsClassifier(n_neighbors=5)
]

# Number of bootstrap samples
n_bootstrap = 10

# Create an empty list to store models
trained_classifiers = []

# Generate bootstrap samples and train each model
for clf in classifiers:
    clf_bootstrap_models = []
    for _ in range(n_bootstrap):
        # Generate bootstrap sample (with replacement)
        X_resampled, y_resampled = resample(X_train, y_train, n_samples=X_train.shape[0], random_state=42)
        clf_clone = clf.__class__()  # Create a fresh clone of the classifier
        clf_clone.fit(X_resampled, y_resampled)
        clf_bootstrap_models.append(clf_clone)
    
    # Append the trained bootstrap models to the list of classifiers
    trained_classifiers.append(clf_bootstrap_models)

# Function to aggregate predictions using majority voting

def bagging_predict(X):
    predictions = []
    
    for clf_bootstrap_models in trained_classifiers:
        clf_preds = np.zeros((X.shape[0], len(clf_bootstrap_models)))
        
        for idx, model in enumerate(clf_bootstrap_models):
            clf_preds[:, idx] = model.predict(X)
        
        # Average predictions for each classifier group
        avg_pred = np.mean(clf_preds, axis=1)
        # Convert to binary predictions using threshold
        binary_pred = (avg_pred >= 0.5).astype(int)
        predictions.append(binary_pred)
    
    # Final ensemble prediction
    final_pred = np.mean(predictions, axis=0)
    return (final_pred >= 0.5).astype(int)

# Apply bagging prediction
final_predictions = bagging_predict(X_valid)

# Evaluate final predictions
accuracy = np.mean(final_predictions == y_valid)
bagging_mcc = matthews_corrcoef(y_valid, final_predictions)
print(f"Bagging Accuracy: {accuracy:.4f}")
print(f"Bagging MCC: {bagging_mcc:.4f}")

#* 4. Fine-tuned FINBERT *#
from transformers import BertTokenizer, BertForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch import nn

# Load the pre-trained 'ProsusAI/finbert' model and tokenizer
model_name = 'ProsusAI/finbert'
tokenizer = BertTokenizer.from_pretrained(model_name)

finbert = BertForSequenceClassification.from_pretrained(model_name)  # Binary classification
finbert.config.num_labels = 2
finbert.num_labels = 2

# Add dropout to the classifier for better regularization
# First apply the base model modifications
dropout_prob = 0.5
finbert.classifier = nn.Sequential(
    nn.Dropout(dropout_prob),
    nn.Linear(finbert.config.hidden_size, finbert.config.num_labels)
)

# Configure PEFT with LoRA specifically for binary classification
# Modify target modules but don't include the classifier in modules_to_save
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.5,
    bias="none",
    target_modules=["query", "key", "value", "output.dense"],
    # Don't include classifier here to avoid the AttributeError
    # We'll manually freeze/unfreeze layers instead
)

# Wrap the model with LoRA
finbert = get_peft_model(finbert, peft_config)

# Manually ensure the classifier parameters are trainable
# This replaces the need for modules_to_save
for param in finbert.classifier.parameters():
    param.requires_grad = True

finbert.print_trainable_parameters()

# set up data loader
from datasets import Dataset

acl_train_ds = Dataset.from_pandas(acl_train_df)
acl_valid_ds = Dataset.from_pandas(acl_valid_df)

def tokenize_function(example):
    return tokenizer(
        example["text"], 
        padding=True,            # Padding handled dynamically by the DataCollator
        truncation=True,
        max_length=128
    )

acl_train_token = acl_train_ds.map(tokenize_function, batched=True)
acl_valid_token = acl_valid_ds.map(tokenize_function, batched=True)

acl_train_token = acl_train_token.rename_column("gold", "label")
acl_valid_token = acl_valid_token.rename_column("gold", "label")

acl_train_token.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
acl_valid_token.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

train_dataloader = DataLoader(acl_train_token, batch_size=16, shuffle=True, collate_fn=data_collator)
valid_dataloader = DataLoader(acl_valid_token, batch_size=16, shuffle=False, collate_fn=data_collator)

# train with torch
import torch
from torch import nn
from transformers import AdamW
from tqdm import tqdm
from transformers import get_scheduler

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert.to(device)
print(device)

# set up optimizer
optimizer = AdamW(finbert.parameters(), lr=1e-5, weight_decay=0.1)  # Reduced learning rate and increased weight decay

# set up loss function
# Add class weights to handle class imbalance if needed
class_weights = torch.tensor([1.0, 1.0], device=device)  # Adjust based on class distribution
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Add early stopping mechanism (now based on MCC instead of accuracy)
best_val_mcc = -1.0  # MCC ranges from -1 to 1
best_val_accuracy = 0
patience = 2
early_stop_counter = 0
best_model_state = None

# Add learning rate scheduler for better convergence
num_epochs = 5  # Increase epochs since we have early stopping
lr_scheduler = get_scheduler(
    "cosine",  # Use cosine schedule for better convergence
    optimizer=optimizer, 
    num_warmup_steps=int(0.1 * len(train_dataloader) * num_epochs),  # 10% warmup
    num_training_steps=len(train_dataloader) * num_epochs
)

# torch training loop with regularization techniques
for epoch in range(num_epochs):
    finbert.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Enable dropout during training (it's enabled by default in train mode)
        outputs = finbert(**batch)
        loss = outputs.loss
        
        # Add L2 regularization if needed
        # for param in finbert.parameters():
        #    loss += 0.01 * torch.sum(param ** 2)
            
        total_loss += loss.item()

        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(finbert.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Avg training loss: {avg_train_loss:.4f}")

    # Validation
    finbert.eval()
    correct, total = 0, 0
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = finbert(**batch)
            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            
            # Collect predictions and labels for MCC calculation
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    val_accuracy = correct / total
    val_mcc = matthews_corrcoef(all_labels, all_preds)
    avg_val_loss = val_loss / len(valid_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, MCC: {val_mcc:.4f}")
    
    # Early stopping check - now based on MCC
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_val_accuracy = val_accuracy
        early_stop_counter = 0
        # Save best model state
        best_model_state = {k: v.cpu().clone() for k, v in finbert.state_dict().items()}
    else:
        early_stop_counter += 1
        
    if early_stop_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

# Load the best model state before saving
if best_model_state is not None:
    finbert.load_state_dict(best_model_state)
    print(f"Loaded best model with validation MCC: {best_val_mcc:.10f}, Accuracy: {best_val_accuracy:.10f}")

# Evaluate the original FinBERT (before LoRA fine-tuning)
print("\n=== Evaluating Original FinBERT Model ===")
original_finbert.to(device)

# Evaluation function for models
from sklearn.metrics import matthews_corrcoef

def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            
            # Collect predictions and labels for MCC calculation
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = correct / total
    avg_loss = val_loss / len(dataloader)
    mcc = matthews_corrcoef(all_labels, all_preds)
    return accuracy, avg_loss, mcc

# save the fine-tuned model
finbert.save_pretrained("finbert-lora-semantic")
tokenizer.save_pretrained("finbert-lora-semantic")

# Evaluate original FinBERT# Evaluate the original FinBERT (before LoRA fine-tuning)
original_finbert = BertForSequenceClassification.from_pretrained(model_name)  # Binary classification
original_finbert.config.num_labels = 2
original_finbert.num_labels = 2
dropout_prob = 0.5
original_finbert.classifier = nn.Sequential(
    nn.Dropout(dropout_prob),
    nn.Linear(original_finbert.config.hidden_size, original_finbert.config.num_labels)
)
original_finbert.to(device)
original_accuracy, original_loss, original_mcc = evaluate_model(original_finbert, valid_dataloader, device)

# Evaluate the fine-tuned FinBERT
finbert_accuracy, finbert_loss, finbert_mcc = evaluate_model(finbert, valid_dataloader, device)

print("\n=== Model Comparison ===")
print(f"Original FinBERT - Validation Loss: {original_loss:.4f}, Accuracy: {original_accuracy:.4f}, MCC: {original_mcc:.4f}")
print(f"Fine-tuned FinBERT - Accuracy: {finbert_accuracy:.4f}, MCC: {finbert_mcc:.4f}")
