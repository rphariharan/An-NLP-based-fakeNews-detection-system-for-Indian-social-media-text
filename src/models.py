from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_naive_bayes(X_train, y_train):
    """
    Train a Naive Bayes (MultinomialNB) model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine (SVM) model.
    """
    # Using linear kernel as it is usually highly effective for text classification
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model", plot_cm=False, save_dir=None):
    """
    Evaluate the model using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    # Using pos_label='Fake' assuming binary classification with labels 'Fake' and 'Real'
    # If using string labels, pos_label must be explicitly handled.
    # Alternatively, use macro/micro average if multclass.
    # Let's use macro/weighted or assume 'Fake' is the positive class.
    
    labels = list(set(y_test))
    # We will use weighted average to handle string labels natively without specifying a pos_label
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    }
    
    print(f"\n--- Evaluation Results for {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    if plot_cm:
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
            # plt.close() # Close to prevent showing automatically if running as script
        else:
            plt.show()
            
    return metrics

def compare_models(metrics_list):
    """
    Display comparison of model performance in a table format.
    """
    df_comparison = pd.DataFrame(metrics_list)
    print("\n=== Model Comparison ===")
    
    # Formatting to string with 4 decimal places for clean display
    format_dict = {
        'Accuracy': lambda x: f"{x:.4f}", 
        'Precision': lambda x: f"{x:.4f}", 
        'Recall': lambda x: f"{x:.4f}", 
        'F1-score': lambda x: f"{x:.4f}"
    }
    
    # Assuming pandas formatting
    print(df_comparison.to_string(index=False, formatters=format_dict))
    
    return df_comparison
