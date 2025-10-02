"""
Week 5 - Binary Step Predictor (Simplified Approach)
=====================================================

Predicts whether a prompt can be accelerated (use 20 steps) or needs full quality (30 steps).

Binary Classification:
    Input: CLIP text embedding (768-dim)
    Output: 0 (needs 30 steps) or 1 (can use 20 steps)

Decision Rule:
    if predict == 1: use 20 steps (~33% speedup)
    else: use 30 steps (full quality)

This is simpler than regression and should work better with limited data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    TRAIN_CSV = "data/training_data_012.csv"
    EMBEDDINGS_CACHE = "data/clip_embeddings.pt"
    
    # Binary threshold: optimal_steps < this value = can accelerate
    ACCELERATION_THRESHOLD = 25
    ACCELERATED_STEPS = 20  # What steps to use when accelerating
    BASELINE_STEPS = 30     # What steps to use for full quality
    
    # Model
    EMBEDDING_DIM = 768
    HIDDEN_DIMS = [256, 128]  # Simpler than regression
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3  # Higher LR for simpler problem
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    VAL_SPLIT = 0.2
    SEED = 42
    
    # Paths
    MODEL_DIR = "models"
    RESULTS_DIR = "results/week5_binary_results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATASET
# ============================================================================

class BinaryStepDataset(Dataset):
    """Dataset for binary classification"""
    
    def __init__(self, prompts, can_accelerate, embeddings):
        self.prompts = prompts
        self.can_accelerate = can_accelerate
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'can_accelerate': torch.tensor(self.can_accelerate[idx], dtype=torch.float32),
            'prompt': self.prompts[idx]
        }

# ============================================================================
# MODEL
# ============================================================================

class BinaryStepPredictor(nn.Module):
    """Binary classifier for step prediction"""
    
    def __init__(self, input_dim=768, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Binary output with sigmoid
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        embeddings = batch['embedding'].to(device)
        labels = batch['can_accelerate'].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            labels = batch['can_accelerate'].to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, df, embeddings, device):
    """Comprehensive evaluation"""
    model.eval()
    
    # Get predictions
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for emb in embeddings:
            prob = model(emb.unsqueeze(0).to(device))
            probabilities.append(prob.item())
            predictions.append(1 if prob.item() > 0.5 else 0)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Ground truth
    can_accelerate = (df['optimal_steps'] < Config.ACCELERATION_THRESHOLD).astype(int).values
    
    # Calculate metrics
    correct = (predictions == can_accelerate).sum()
    accuracy = correct / len(predictions)
    
    # Per-class metrics
    tp = ((predictions == 1) & (can_accelerate == 1)).sum()  # True positives
    fp = ((predictions == 1) & (can_accelerate == 0)).sum()  # False positives
    tn = ((predictions == 0) & (can_accelerate == 0)).sum()  # True negatives
    fn = ((predictions == 0) & (can_accelerate == 1)).sum()  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Speedup calculation
    predicted_steps = np.where(predictions == 1, Config.ACCELERATED_STEPS, Config.BASELINE_STEPS)
    actual_optimal_steps = df['optimal_steps'].values
    
    predicted_speedup = ((Config.BASELINE_STEPS - predicted_steps.mean()) / Config.BASELINE_STEPS) * 100
    actual_speedup = ((Config.BASELINE_STEPS - actual_optimal_steps.mean()) / Config.BASELINE_STEPS) * 100
    
    # Quality preservation (for prompts we accelerate)
    accelerated_mask = predictions == 1
    if accelerated_mask.sum() > 0:
        accelerated_lpips = df.loc[accelerated_mask, 'lpips_at_20'].values
        mean_lpips_accelerated = accelerated_lpips.mean()
        quality_maintained = (accelerated_lpips < 0.12).sum() / len(accelerated_lpips)
    else:
        mean_lpips_accelerated = 0
        quality_maintained = 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'predicted_speedup': predicted_speedup,
        'actual_speedup': actual_speedup,
        'mean_lpips_accelerated': mean_lpips_accelerated,
        'quality_maintained': quality_maintained,
        'predictions': predictions,
        'probabilities': probabilities,
        'ground_truth': can_accelerate
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves to {save_path}")

def plot_confusion_matrix(results, save_path):
    """Plot confusion matrix"""
    cm = np.array([[results['tn'], results['fp']], 
                   [results['fn'], results['tp']]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Needs 30 steps', 'Can accelerate'])
    ax.set_yticklabels(['Needs 30 steps', 'Can accelerate'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=20, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix to {save_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("WEEK 5 - BINARY STEP PREDICTOR TRAINING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {Config.DEVICE}")
    print()
    
    # Setup
    Path(Config.MODEL_DIR).mkdir(exist_ok=True)
    Path(Config.RESULTS_DIR).mkdir(exist_ok=True)
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(Config.TRAIN_CSV)
    print(f"✅ Loaded {len(df)} prompts")
    
    # Create binary labels
    df['can_accelerate'] = (df['optimal_steps'] < Config.ACCELERATION_THRESHOLD).astype(int)
    
    print(f"\n📊 Binary distribution:")
    print(df['can_accelerate'].value_counts())
    print(f"   Class 0 (needs 30): {(df['can_accelerate']==0).sum()} ({(df['can_accelerate']==0).mean()*100:.1f}%)")
    print(f"   Class 1 (can accel): {(df['can_accelerate']==1).sum()} ({(df['can_accelerate']==1).mean()*100:.1f}%)")
    print()
    
    # Load embeddings
    print("📂 Loading CLIP embeddings...")
    cache = torch.load(Config.EMBEDDINGS_CACHE)
    embeddings = cache['embeddings']
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    print()
    
    # Train/val split
    print("🔀 Splitting into train/validation...")
    prompts = df['prompt'].tolist()
    labels = df['can_accelerate'].values
    
    indices = np.arange(len(prompts))
    train_idx, val_idx = train_test_split(
        indices, test_size=Config.VAL_SPLIT, random_state=Config.SEED, stratify=labels
    )
    
    train_dataset = BinaryStepDataset(
        [prompts[i] for i in train_idx],
        labels[train_idx],
        embeddings[train_idx]
    )
    
    val_dataset = BinaryStepDataset(
        [prompts[i] for i in val_idx],
        labels[val_idx],
        embeddings[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"✅ Train: {len(train_dataset)} samples")
    print(f"✅ Val: {len(val_dataset)} samples")
    print()
    
    # Initialize model
    print("🏗️  Initializing model...")
    model = BinaryStepPredictor(
        input_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT
    )
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")
    print()
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("🚀 Starting training...")
    print("="*80)
    print()
    
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        
        # Validate
        val_loss, val_acc, _, _, _ = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = Path(Config.MODEL_DIR) / "binary_step_predictor_v1.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'embedding_dim': Config.EMBEDDING_DIM,
                    'hidden_dims': Config.HIDDEN_DIMS,
                    'dropout': Config.DROPOUT,
                    'acceleration_threshold': Config.ACCELERATION_THRESHOLD,
                    'accelerated_steps': Config.ACCELERATED_STEPS,
                    'baseline_steps': Config.BASELINE_STEPS
                }
            }, model_path)
            print(f"   💾 Saved best model (Val Acc: {val_acc:.3f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
        
        print()
    
    print("="*80)
    print(f"✅ Training complete! Best validation accuracy: {best_val_acc:.3f}")
    print()
    
    # Plot training curves
    curves_path = Path(Config.RESULTS_DIR) / "binary_training_curves.png"
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
    print()
    
    # Load best model and evaluate
    print("📊 Final evaluation...")
    checkpoint = torch.load(
        Path(Config.MODEL_DIR) / "binary_step_predictor_v1.pth",
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    results = evaluate_model(model, df, embeddings, Config.DEVICE)
    
    # Print results
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print()
    print(f"📊 Classification Metrics:")
    print(f"   Accuracy: {results['accuracy']:.1%}")
    print(f"   Precision: {results['precision']:.1%}")
    print(f"   Recall: {results['recall']:.1%}")
    print(f"   F1 Score: {results['f1']:.3f}")
    print()
    print(f"📊 Confusion Matrix:")
    print(f"   True Negatives:  {results['tn']}")
    print(f"   False Positives: {results['fp']}")
    print(f"   False Negatives: {results['fn']}")
    print(f"   True Positives:  {results['tp']}")
    print()
    print(f"🚀 Speedup Analysis:")
    print(f"   Predicted speedup: {results['predicted_speedup']:.1f}%")
    print(f"   Actual optimal speedup: {results['actual_speedup']:.1f}%")
    print()
    print(f"✨ Quality Preservation:")
    print(f"   Mean LPIPS (accelerated): {results['mean_lpips_accelerated']:.4f}")
    print(f"   Quality maintained: {results['quality_maintained']:.1%}")
    print()
    
    # Plot confusion matrix
    cm_path = Path(Config.RESULTS_DIR) / "binary_confusion_matrix.png"
    plot_confusion_matrix(results, cm_path)
    print()
    
    # Save report
    report_path = Path(Config.RESULTS_DIR) / "binary_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("BINARY STEP PREDICTOR VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.3f}\n\n")
        f.write(f"Classification Metrics:\n")
        f.write(f"  Accuracy: {results['accuracy']:.1%}\n")
        f.write(f"  Precision: {results['precision']:.1%}\n")
        f.write(f"  Recall: {results['recall']:.1%}\n")
        f.write(f"  F1 Score: {results['f1']:.3f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN: {results['tn']}, FP: {results['fp']}\n")
        f.write(f"  FN: {results['fn']}, TP: {results['tp']}\n\n")
        f.write(f"Speedup: {results['predicted_speedup']:.1f}%\n")
        f.write(f"Quality (LPIPS): {results['mean_lpips_accelerated']:.4f}\n")
        f.write(f"Quality maintained: {results['quality_maintained']:.1%}\n")
    
    print(f"✅ Saved report to {report_path}")
    print()
    
    print("="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print()
    print("Outputs:")
    print(f"   📦 Model: {Path(Config.MODEL_DIR) / 'binary_step_predictor_v1.pth'}")
    print(f"   📊 Training curves: {curves_path}")
    print(f"   📈 Confusion matrix: {cm_path}")
    print(f"   📄 Report: {report_path}")
    print()

if __name__ == "__main__":
    main()