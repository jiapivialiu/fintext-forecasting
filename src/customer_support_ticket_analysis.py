import os
import kagglehub
import pandas as pd

# Get path to the CSV data
download_path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
csv_files = [file for file in os.listdir(download_path) if file.endswith('.csv')]
file_path = os.path.join(download_path, csv_files[0])

# Download the customer support ticket dataset from Kaggle
df = pd.read_csv(file_path)

# Display dataset information
print(f"Dataset shape: {df.shape}")
print("First few rows of the dataset:")
df.head()

#* Customer Demographics *#
# Plot the histogram of customer age
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(df['customer_age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Compute percentage of customers' genders
gender_percentages = df['Customer Gender'].value_counts(normalize=True) * 100
print("\nGender Distribution:")
print(gender_percentages.round(2).to_string())

#* Product information *#
# Count and display the frequency of each product purchased
product_counts = df['Product Purchased'].value_counts()
print("\nProduct Purchase Distribution:")
print(product_counts.head(10).to_string())

# Create a bar plot of product purchases
plt.figure(figsize=(12, 6))
plt.bar(product_counts.index, product_counts.values)
plt.title('Distribution of Products Purchased')
plt.xlabel('Product')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Date of Purchase
# Convert 'Date of Purchase' to datetime
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'])

# Create time series plot of purchases
plt.figure(figsize=(12, 6))
df['Date of Purchase'].value_counts().sort_index().plot(kind='line')
plt.title('Purchase Timeline')
plt.xlabel('Date')
plt.ylabel('Number of Purchases')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create scatter plot of products over time
plt.figure(figsize=(15, 8))
plt.scatter(df['Date of Purchase'], df['Product Purchased'], alpha=0.5)
plt.title('Products Purchased Over Time')
plt.xlabel('Date of Purchase')
plt.ylabel('Product')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Summarize ticket types and ticket subjects
# Create cross-tabulation of ticket types and subjects
ticket_distribution = pd.crosstab(df['Ticket Type'], df['Ticket Subject'])
print("\nTicket Type and Subject Distribution:")
print(ticket_distribution)
# Create bar plot of ticket types and subjects
# Using colorblind friendly colors from ColorBrewer
colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
          '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
ticket_distribution.plot(kind='bar', figsize=(12, 6), stacked=True, color=colors)
plt.title('Distribution of Ticket Types by Subject')
plt.xlabel('Ticket Type')
plt.ylabel('Count')
plt.legend(title='Ticket Subject', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# summarise ticket status
ticket_status_summary = df['Ticket Status'].value_counts()
print("\nTicket Status Summary:")
print(ticket_status_summary)

# Customer Satisfaction Rating versus ticket status
# Create a cross-tabulation of ticket status and customer satisfaction rating
ticket_status_rating = pd.crosstab(df['Ticket Status'], df['Customer Satisfaction Rating'])
print("\nTicket Status vs Customer Satisfaction Rating:")
print(ticket_status_rating)


# Display summary statistics of resolution duration
print("\nResolution Duration Statistics (hours):")
print(df['Resolution_Duration'].describe().round(2))

# Create histogram of resolution duration
plt.figure(figsize=(10, 6))
plt.hist(df['Resolution_Duration'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Resolution Duration')
plt.xlabel('Duration (hours)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Summarize Customer Satisfaction Rating
satisfaction_summary = df['Customer Satisfaction Rating'].describe()
print("\nCustomer Satisfaction Rating Summary:")
print(satisfaction_summary)

# Create histogram of satisfaction ratings
plt.figure(figsize=(10, 6))
plt.bar(pd.factorize(df['Customer Satisfaction Rating'])[1], df['Customer Satisfaction Rating'].value_counts().sort_index(), color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Satisfaction Ratings')
plt.xlabel('Satisfaction Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate percentage distribution of ratings
satisfaction_dist = df['Customer Satisfaction Rating'].value_counts(normalize=True) * 100
print("\nDistribution of Satisfaction Ratings (%):")
print(satisfaction_dist.sort_index().round(2))

#* Data Preprocessing *#

# drop rows with missing values in customer satisfaction rating
df = df.dropna(subset=['Customer Satisfaction Rating'])

# process date data
# Convert First Response Time and Time to Resolution to datetime
df['First Response Time'] = pd.to_datetime(df['First Response Time'], format='%Y-%m-%d %H:%M:%S')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], format='%Y-%m-%d %H:%M:%S')

# Calculate time difference in hours
df['Resolution_Duration'] = (df['Time to Resolution'] - df['First Response Time']).dt.total_seconds() / 3600

# non-text columns
df_nontext = df[['Customer Age', 'Customer Gender', 'Ticket Priority', 'Resolution_Duration', 'Customer Satisfaction Rating']]
# Factorize categorical variables
df_nontext['Customer Gender'] = pd.factorize(df_nontext['Customer Gender'])[0]
df_nontext['Ticket Priority'] = pd.factorize(df_nontext['Ticket Priority'])[0]
df_nontext['Customer Satisfaction Rating'] = pd.factorize(df_nontext['Customer Satisfaction Rating'])[0]

#* Text Embedding *#
# text embedding for Product Purchased, Ticket Type, Ticket Subject,	Ticket Description,	Ticket Status,	Resolution,	Ticket Priority,	Ticket Channel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Encode the text columns
text_columns = ['Product Purchased', 'Ticket Type', 'Ticket Subject', 'Ticket Description', 'Ticket Status', 'Resolution', 'Ticket Channel']
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Encode categorical columns
for col in text_columns:
    df[col] = label_encoder.fit_transform(df[col])
# Encode the text columns
embeddings = []
for col in text_columns:
    embedding = model.encode(df[col].astype(str).tolist())
    embeddings.append(embedding)
# Concatenate the embeddings
embeddings = np.concatenate(embeddings, axis=1)
# Perform PCA
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)
# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_embeddings, columns=['PCA1', 'PCA2'])
# Add the original labels to the PCA DataFrame
pca_df['Product Purchased'] = df['Product Purchased']
# Plot the PCA results
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Product Purchased', data=pca_df, palette='viridis', alpha=0.7)
plt.title('PCA of Text Embeddings')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Product Purchased', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# combine the embeddings with the non-text columns
# Convert the embeddings to a DataFrame with meaningful column names
embedded_df = pd.DataFrame(embeddings, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])

# Reset index of df_nontext to ensure proper concatenation
df_nontext = df_nontext.reset_index(drop=True)

# Combine the embeddings with non-text features
combined_df = pd.concat([df_nontext, embedded_df], axis=1)

#* Modeling *#
# Build a classification model to predict ticket rating categories
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Check current target distribution
print("\nCustomer Satisfaction Rating Distribution:")
print(combined_df['Customer Satisfaction Rating'].value_counts())

# Prepare features (X) and target (y) variables
X = combined_df.drop('Customer Satisfaction Rating', axis=1)
y = combined_df['Customer Satisfaction Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Train a Random Forest model
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {accuracy_rf:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Try Gradient Boosting for comparison
print("\nTraining Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Evaluate Gradient Boosting model
y_pred_gb = gb_model.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"\nGradient Boosting Accuracy: {accuracy_gb:.4f}")
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

# compute comparison confusion matrix
plt.figure(figsize=(10, 8))
cm_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix - Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Cross-validation to ensure model robustness
print("\nPerforming 5-fold cross-validation...")
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_gb = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"Random Forest CV Scores: {cv_scores_rf}")
print(f"Random Forest Average CV Score: {cv_scores_rf.mean():.4f}")
print(f"Gradient Boosting CV Scores: {cv_scores_gb}")
print(f"Gradient Boosting Average CV Score: {cv_scores_gb.mean():.4f}")

# Save the best model for future use
import joblib
if accuracy_rf >= accuracy_gb:
    print("\nSaving Random Forest model as it performed better")
    joblib.dump(rf_model, 'ticket_rating_model.pkl')
    best_model = rf_model
else:
    print("\nSaving Gradient Boosting model as it performed better")
    joblib.dump(gb_model, 'ticket_rating_model.pkl')
    best_model = gb_model

#* LLM *#
import tensorflow as tf
import os

# Try to limit GPU memory growth to avoid memory errors
try:
    # Set memory growth and limit TensorFlow GPU memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("GPUs found, setting memory growth...")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally limit GPU memory to avoid out-of-memory errors
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        print(f"GPU configuration successful: {len(gpus)} GPU(s) available")
    else:
        print("No GPUs found, using CPU instead")
except Exception as e:
    print(f"Error configuring GPU: {e}")
    print("Falling back to CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

# Import other necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

print("\n" + "="*50)
print("Deep Learning Models for Customer Satisfaction Prediction")
print("="*50)

# combine text and non-text features - use a smaller subset of features to reduce complexity
# Select only important columns to reduce dimensionality
df_llm = pd.concat([
    df_nontext[['Customer Age', 'Customer Gender', 'Resolution_Duration', 'Customer Satisfaction Rating']], 
    df[['Product Purchased', 'Ticket Type', 'Ticket Subject']]
], axis=1)

y = df_llm['Customer Satisfaction Rating']
X = df_llm.drop('Customer Satisfaction Rating', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=814, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get unique classes and their mapping
unique_classes = sorted(y.unique())
n_classes = len(unique_classes)
print(f"Number of satisfaction rating classes: {n_classes}")
class_mapping = {class_val: i for i, class_val in enumerate(unique_classes)}
reverse_mapping = {i: class_val for i, class_val in enumerate(unique_classes)}

# Convert target to categorical for neural network
y_train_cat = to_categorical([class_mapping[val] for val in y_train])
y_test_cat = to_categorical([class_mapping[val] for val in y_test])

# Neural Network model with TensorFlow/Keras
print("\nBuilding and training Neural Network model...")

# Wrap the model building and training in a try-except block
try:
    # Define a simpler model architecture
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    
    # Compile model with a more standard optimizer
    nn_model.compile(
        optimizer='adam',  # Standard Adam optimizer instead of AdamW
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary to check complexity
    nn_model.summary()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Use a smaller batch size to reduce memory usage
    batch_size = 32  # Reduced from 64
    
    # Train the model with fewer epochs initially
    nn_history = nn_model.fit(
        X_train_scaled, 
        y_train_cat,
        epochs=15,  # Reduced from 20
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate neural network model
    y_pred_nn_prob = nn_model.predict(X_test_scaled)
    y_pred_nn = np.array([reverse_mapping[np.argmax(pred)] for pred in y_pred_nn_prob])
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    print(f"\nNeural Network Accuracy: {accuracy_nn:.4f}")
    print("\nNeural Network Classification Report:")
    print(classification_report(y_test, y_pred_nn))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nn_history.history['accuracy'])
    plt.plot(nn_history.history['val_accuracy'])
    plt.title('Neural Network Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(nn_history.history['loss'])
    plt.plot(nn_history.history['val_loss'])
    plt.title('Neural Network Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix for neural network
    plt.figure(figsize=(10, 8))
    cm_nn = confusion_matrix(y_test, y_pred_nn)
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title('Confusion Matrix - Neural Network')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error during neural network training: {e}")
    print("Falling back to a simpler model...")
    
    # Try an even simpler model as fallback
    try:
        # Very simple model with minimal layers
        simple_model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(n_classes, activation='softmax')
        ])
        
        simple_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with minimal parameters
        simple_history = simple_model.fit(
            X_train_scaled, 
            y_train_cat,
            epochs=10,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate the simple model
        y_pred_simple = np.array([reverse_mapping[np.argmax(pred)] for pred in 
                                 simple_model.predict(X_test_scaled)])
        accuracy_nn = accuracy_score(y_test, y_pred_simple)
        print(f"\nSimple Neural Network Accuracy: {accuracy_nn:.4f}")
        
        # Assign the simple model to be our neural network model
        nn_model = simple_model
        nn_history = simple_history
    except Exception as e2:
        print(f"Simple model also failed: {e2}")
        print("Skipping neural network approach entirely.")
        accuracy_nn = 0.0

# Transformer-based model approach
print("\n" + "="*50)
print("Transformer-based Language Model for Satisfaction Rating Prediction")
print("="*50)

# Function to prepare text data from the ticket data
def prepare_text_data(df):
    # Combine relevant text columns into a single text field
    return df['Ticket Description'] + " " + df['Resolution'] + " Product: " + df['Product Purchased']

# Get original text data from the dataframe for a sample (adjust as needed)
# For demonstration, we'll sample some data to make training faster
# In a production scenario, you might use the full dataset
sample_size = min(1000, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Prepare text and labels
texts = prepare_text_data(df_sample).reset_index(drop=True)
labels = pd.factorize(df_sample['Customer Satisfaction Rating'].reset_index(drop=True))[0]

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Try to use transformer model if transformers library and GPU are available
try:
    # Load pre-trained tokenizer and model
    model_name = "distilbert-base-uncased"  # A smaller, faster BERT variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the texts
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)
    
    # Create PyTorch datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_labels
    })
    
    # Load the model
    lm_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(set(labels))
    )
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lm_model.to(device)
    
    # Training arguments
    batch_size = 16
    training_args = TrainingArguments(
        output_dir='./transformer_results',
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./transformer_logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize Trainer
    trainer = Trainer(
        model=lm_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train the model
    print("\nTraining transformer model...")
    trainer.train()
    
    # Evaluate the model
    print("\nEvaluating transformer model...")
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    
    # Calculate and display metrics
    accuracy_lm = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='weighted')
    
    print(f"Transformer Model Accuracy: {accuracy_lm:.4f}")
    print(f"Transformer Model F1 Score: {f1:.4f}")
    print("\nTransformer Model Classification Report:")
    print(classification_report(test_labels, preds))

except Exception as e:
    print(f"\nSkipping transformer model due to error: {str(e)}")
    print("This could be due to missing dependencies or hardware limitations.")

# Model comparison 
print("\n" + "="*50)
print("Model Performance Comparison")
print("="*50)
models = ["Random Forest", "Gradient Boosting", "Neural Network"]
accuracies = [accuracy_rf, accuracy_gb, accuracy_nn]

# Add transformer model if it was successfully trained
try:
    if 'accuracy_lm' in locals():
        models.append("Transformer LLM")
        accuracies.append(accuracy_lm)
except:
    pass

# Create bar chart comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple'][:len(models)])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.title('Model Accuracy Comparison for Customer Satisfaction Rating Prediction')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Determine the best model
best_model_name = models[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\nBest performing model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Save the neural network model if requested
save_nn = input("\nDo you want to save the neural network model? (y/n): ")
if save_nn.lower() == 'y':
    nn_model.save('ticket_rating_neural_network.h5')
    print("Neural network model saved as 'ticket_rating_neural_network.h5'")

print("\nCustomer Support Ticket Analysis and Rating Prediction Complete!")
