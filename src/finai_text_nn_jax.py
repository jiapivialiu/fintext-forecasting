# Neural nets on embedded text with JAX
from datasets import load_dataset

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

# embedding model
from sentence_transformers import SentenceTransformer

# Initialize the model (using a compact and efficient model)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get sentence embeddings
def get_sbert_embeddings(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# Combine embedded text features with non-text features
import numpy as np
texts = bigdata_train_df['text'].tolist()
X_text_embeddings = get_sbert_embeddings(texts)
X_text_embeddings = np.array(X_text_embeddings, dtype=np.float32)
bigdata_train_embedded = np.concatenate([bigdata_train_df['gold'], X_text_embeddings], axis=1)

# Convert the combined data to JAX arrays if needed
import jax.numpy as jnp
bigdata_train_embedded_jax = jnp.array(bigdata_train_embedded, dtype=jnp.float32)
bigdata_train_embedded_jax.shape

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import flax
from flax import linen as nn
from jax import random

# initialize random parameters
def init_params(layer_sizes, key):
    """Initialize parameters for a simple MLP model."""
    params = []
    for i in range(len(layer_sizes) - 1):
        # Initialize weights with a small random value and biases as zeros
        w_key, b_key = random.split(key)
        w = random.normal(w_key, (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        params.append((w, b))
    return params

# Define the neural nets with enhancements
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return jnp.where(x > 0, x, alpha * x)

def mlp(params, X, dropout_key=None, dropout_rate=0.2):
    """A feedforward MLP with dropout."""
    for i, (w, b) in enumerate(params[:-1]):
        X = jnp.dot(X, w) + b
        X = leaky_relu(X)  # Use LeakyReLU activation
        if dropout_key is not None:
            # Apply dropout during training
            dropout_key, subkey = random.split(dropout_key)
            mask = random.bernoulli(subkey, p=1 - dropout_rate, shape=X.shape)
            X = X * mask / (1 - dropout_rate)
    w, b = params[-1]
    return jnp.dot(X, w) + b  # Linear output layer

# Update the loss function to binary cross-entropy
def binary_cross_entropy_loss(params, X, y):
    """Compute the binary cross-entropy loss."""
    logits = mlp(params, X)
    preds = jax.nn.sigmoid(logits)  # Apply sigmoid for binary classification
    return -jnp.mean(y * jnp.log(preds + 1e-8) + (1 - y) * jnp.log(1 - preds + 1e-8))

# Compute gradients for the updated loss function
grad_loss_fn = grad(binary_cross_entropy_loss)

# Update the training step to include dropout
@jit
def train_step(params, X, y, key, learning_rate=0.001, dropout_rate=0.2):
    """Perform one step of gradient descent with dropout."""
    dropout_key, subkey = random.split(key)
    grads = grad_loss_fn(params, X, y)
    new_params = [(w - learning_rate * dw, b - learning_rate * db) 
                  for (w, b), (dw, db) in zip(params, grads)]
    return new_params, dropout_key

# Training data (replace with your own)
X_train = bigdata_train_embedded_jax[:, 1:]
print(X_train[:10,:])
print(X_train.shape)
y_train = bigdata_train_embedded_jax[:, 0]
print(y_train[:10])
print(y_train.shape)

# Training loop with dropout
key = random.PRNGKey(0)
layer_sizes = [X_train.shape[1], 128, 64, 32, 1]  # Enhanced layer sizes
params = init_params(layer_sizes, key)
dropout_key = random.PRNGKey(1)

num_epochs = 3000
for epoch in range(num_epochs):
    params, dropout_key = train_step(params, X_train, y_train, dropout_key, learning_rate=0.005, dropout_rate=0.2)
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        loss = binary_cross_entropy_loss(params, X_train, y_train)
        print(f"Epoch {epoch}, Loss: {loss}")

# Process validation data
valid_texts = bigdata_valid_df['text'].tolist()
X_valid_embeddings = get_sbert_embeddings(valid_texts)
X_valid = jnp.array(X_valid_embeddings, dtype=jnp.float32)
y_valid = jnp.array(bigdata_valid_df['gold'].values, dtype=jnp.float32)

# Evaluate on validation set with sigmoid activation
valid_logits = mlp(params, X_valid)
valid_preds = jax.nn.sigmoid(valid_logits)
valid_loss = binary_cross_entropy_loss(params, X_valid, y_valid)
print(f"Validation Loss: {valid_loss}")

# Calculate accuracy
valid_preds_binary = (valid_preds > 0.5).astype(jnp.float32)
accuracy = jnp.mean(valid_preds_binary == y_valid)
print(f"Validation Accuracy: {accuracy}")
