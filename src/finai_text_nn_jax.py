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

#* 1. Logistic Regression *#
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Prepare data for modelling
y_train = acl_train_embedded[:, 0].reshape(-1, 1)
X_train = acl_train_embedded[:, 1:]
y_valid = acl_valid_embedded[:, 0].reshape(-1, 1)
X_valid = acl_valid_embedded[:, 1:]

# Train logistic regression model
maxiter = 1000
lr_model = LogisticRegression(max_iter=maxiter, random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = lr_model.predict(X_valid)

# Print performance metrics
print("\nLogistic Regression Results:")
print(f"Validation Accuracy: {lr_model.score(X_valid, y_valid):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_valid, y_pred))

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

# Hyperparameter tuning
def tune_hyperparameters():
    best_accuracy = 0.0
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
                    val_accuracy = evaluate(params, X_valid, y_valid, dropout_rate=0.0, train=False)
                    print(f"Validation Accuracy: {val_accuracy:.4f}")
                    
                    # Record result
                    config = {
                        'learning_rate': lr,
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'batch_size': batch_size,
                        'val_accuracy': val_accuracy
                    }
                    results.append(config)
                    
                    # Update best model
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = params
                        best_config = config
    
    print("\n=== Hyperparameter Tuning Results ===")
    for i, res in enumerate(sorted(results, key=lambda x: x['val_accuracy'], reverse=True)):
        print(f"{i+1}. Accuracy: {res['val_accuracy']:.4f} - LR: {res['learning_rate']}, " 
              f"Layers: {res['hidden_layers']}, Dropout: {res['dropout_rate']}, "
              f"Batch Size: {res['batch_size']}")
    
    print(f"\nBest Configuration: {best_config}")
    return best_params, best_config

# Modified evaluation function for hyperparameter tuning
def evaluate(params, X, y, dropout_rate=0.0, train=False) -> float:
    """Evaluate model accuracy with optional dropout"""
    key = random.PRNGKey(99) if train else None
    logits = mlp_forward(params, X, dropout_rate=dropout_rate, train=train, key=key)
    preds = jax.nn.sigmoid(logits)
    binary_preds = (preds > 0.5).astype(jnp.float32)
    accuracy = jnp.mean(binary_preds == y)
    return float(accuracy)

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
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        train_acc = evaluate(params, X_train, y_train, dropout_rate=0.0, train=False)
        val_acc = evaluate(params, X_valid, y_valid, dropout_rate=0.0, train=False)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Final evaluation
final_accuracy = evaluate(params, X_valid, y_valid, dropout_rate=0.0, train=False)
print(f"\nFinal Validation Accuracy: {final_accuracy:.10f}")
