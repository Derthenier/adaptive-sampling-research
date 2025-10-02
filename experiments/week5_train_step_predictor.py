"""
Week 5 - Train Step Predictor Model
====================================

Trains a lightweight MLP to predict optimal step count from prompt text.

Architecture:
    CLIP Text Encoder (frozen) → Prompt Embedding (768-dim)
    MLP Predictor (trainable) → Optimal Steps (15-30)

Training:
    - Dataset: training_data_012.csv (0.12 threshold)
    - Split: 80% train, 20% validation
    - Loss: MSE (regression) or BCE (classification)
    - Validation: Against both 0.10 and 0.12 thresholds

Output:
    - models/step_predictor_v1.pth (best model)
    - results/training_curves.png (visualization)
    - results/validation_report.txt (metrics)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    TRAIN_CSV = "data/training_data_012.csv"  # Train on 0.12 threshold
    VAL_CSV_010 = "data/training_data_010.csv"  # Validate on 0.10
    VAL_CSV_012 = "data/training_data_012.csv"  # Validate on 0.12
    
    # Model
    CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
    EMBEDDING_DIM = 768  # CLIP text encoder output dim
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT = 0.2
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    VAL_SPLIT = 0.2
    SEED = 42
    
    # Output range
    MIN_STEPS = 15
    MAX_STEPS = 30
    
    # Paths
    MODEL_DIR = "models"
    RESULTS_DIR = "results/week5_results"
    EMBEDDINGS_CACHE = "data/clip_embeddings.pt"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATASET
# ============================================================================

class StepPredictorDataset(Dataset):
    """Dataset for step prediction"""
    
    def __init__(self, prompts, optimal_steps, embeddings):
        self.prompts = prompts
        self.optimal_steps = optimal_steps
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'optimal_steps': torch.tensor(self.optimal_steps[idx], dtype=torch.float32),
            'prompt': self.prompts[idx]
        }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class StepPredictor(nn.Module):
    """MLP that predicts optimal steps from CLIP embeddings"""
    
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer - single value (steps)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Store range for clamping
        self.min_steps = Config.MIN_STEPS
        self.max_steps = Config.MAX_STEPS
    
    def forward(self, x):
        # Raw output
        out = self.network(x).squeeze(-1)
        
        # Clamp to valid range [15, 30]
        out = torch.clamp(out, self.min_steps, self.max_steps)
        
        return out

# ============================================================================
# CLIP EMBEDDING EXTRACTION
# ============================================================================

def extract_clip_embeddings(prompts, cache_path=None):
    """Extract CLIP text embeddings for prompts"""
    
    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"📂 Loading cached embeddings from {cache_path}")
        cache = torch.load(cache_path)
        if cache['prompts'] == prompts:
            print(f"✅ Cache valid, loaded {len(prompts)} embeddings")
            return cache['embeddings']
        else:
            print(f"⚠️  Cache invalid (different prompts), regenerating...")
    
    print(f"🔄 Extracting CLIP embeddings for {len(prompts)} prompts...")
    
    # Load CLIP
    tokenizer = CLIPTokenizer.from_pretrained(Config.CLIP_MODEL_NAME)
    text_encoder = CLIPTextModel.from_pretrained(Config.CLIP_MODEL_NAME)
    text_encoder = text_encoder.to(Config.DEVICE)
    text_encoder.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting embeddings"):
            # Tokenize
            inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = text_encoder(**inputs)
            # Use pooled output (CLS token)
            embedding = outputs.pooler_output.cpu()
            embeddings.append(embedding)
    
    embeddings = torch.cat(embeddings, dim=0)
    
    # Cache for future use
    if cache_path:
        print(f"💾 Caching embeddings to {cache_path}")
        torch.save({
            'prompts': prompts,
            'embeddings': embeddings
        }, cache_path)
    
    print(f"✅ Extracted embeddings: shape {embeddings.shape}")
    return embeddings

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in dataloader:
        embeddings = batch['embedding'].to(device)
        steps = batch['optimal_steps'].to(device)
        
        # Forward
        optimizer.zero_grad()
        pred_steps = model(embeddings)
        loss = criterion(pred_steps, steps)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(pred_steps.detach().cpu().numpy())
        targets.extend(steps.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    
    return avg_loss, mae

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            steps = batch['optimal_steps'].to(device)
            
            pred_steps = model(embeddings)
            loss = criterion(pred_steps, steps)
            
            total_loss += loss.item()
            predictions.extend(pred_steps.cpu().numpy())
            targets.extend(steps.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    
    return avg_loss, mae, predictions, targets

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictor(model, df, embeddings, threshold_name, device):
    """Evaluate predictor on dataset"""
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for emb in embeddings:
            pred = model(emb.unsqueeze(0).to(device))
            predictions.append(pred.item())
    
    predictions = np.array(predictions)
    targets = df['optimal_steps'].values
    
    # Round predictions to nearest valid step count
    valid_steps = np.array([15, 18, 20, 22, 25, 30])
    rounded_preds = np.array([
        valid_steps[np.argmin(np.abs(valid_steps - p))]
        for p in predictions
    ])
    
    # Metrics
    mae = np.mean(np.abs(predictions - targets))
    mae_rounded = np.mean(np.abs(rounded_preds - targets))
    exact_match = np.mean(rounded_preds == targets)
    within_1_step = np.mean(np.abs(rounded_preds - targets) <= 3)  # Within one bucket
    
    # Speedup calculation
    baseline_steps = 30
    predicted_speedup = ((baseline_steps - predictions.mean()) / baseline_steps) * 100
    actual_speedup = ((baseline_steps - targets.mean()) / baseline_steps) * 100
    
    results = {
        'threshold': threshold_name,
        'mae': mae,
        'mae_rounded': mae_rounded,
        'exact_match': exact_match,
        'within_1_step': within_1_step,
        'predicted_speedup': predicted_speedup,
        'actual_speedup': actual_speedup,
        'predictions': predictions,
        'rounded_predictions': rounded_preds,
        'targets': targets
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(train_losses, val_losses, train_maes, val_maes, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE curves
    ax2.plot(train_maes, label='Train MAE', linewidth=2)
    ax2.plot(val_maes, label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (steps)', fontsize=12)
    ax2.set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves to {save_path}")

def plot_predictions_vs_targets(results_010, results_012, save_path):
    """Plot predicted vs actual steps for both thresholds"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 0.10 threshold
    ax1.scatter(results_010['targets'], results_010['rounded_predictions'], 
                alpha=0.6, s=100, edgecolors='black', linewidth=1)
    ax1.plot([15, 30], [15, 30], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Optimal Steps', fontsize=12)
    ax1.set_ylabel('Predicted Steps', fontsize=12)
    ax1.set_title(f'Threshold 0.10 (MAE: {results_010["mae_rounded"]:.2f})', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(12, 32)
    ax1.set_ylim(12, 32)
    
    # 0.12 threshold
    ax2.scatter(results_012['targets'], results_012['rounded_predictions'], 
                alpha=0.6, s=100, edgecolors='black', linewidth=1, color='green')
    ax2.plot([15, 30], [15, 30], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Optimal Steps', fontsize=12)
    ax2.set_ylabel('Predicted Steps', fontsize=12)
    ax2.set_title(f'Threshold 0.12 (MAE: {results_012["mae_rounded"]:.2f})', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(12, 32)
    ax2.set_ylim(12, 32)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved prediction scatter plots to {save_path}")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("\n" + "="*80)
    print("WEEK 5 - STEP PREDICTOR TRAINING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {Config.DEVICE}")
    print()
    
    # Create directories
    Path(Config.MODEL_DIR).mkdir(exist_ok=True)
    Path(Config.RESULTS_DIR).mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Load data
    print("📂 Loading training data...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    print(f"✅ Loaded {len(df_train)} prompts from {Config.TRAIN_CSV}")
    print()
    
    # Extract CLIP embeddings
    prompts = df_train['prompt'].tolist()
    embeddings = extract_clip_embeddings(prompts, Config.EMBEDDINGS_CACHE)
    print()
    
    # Prepare data
    optimal_steps = df_train['optimal_steps'].values
    
    # Train/val split
    print("🔀 Splitting into train/validation sets...")
    indices = np.arange(len(prompts))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=Config.VAL_SPLIT, 
        random_state=Config.SEED
    )
    
    train_dataset = StepPredictorDataset(
        [prompts[i] for i in train_idx],
        optimal_steps[train_idx],
        embeddings[train_idx]
    )
    
    val_dataset = StepPredictorDataset(
        [prompts[i] for i in val_idx],
        optimal_steps[val_idx],
        embeddings[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"✅ Train: {len(train_dataset)} samples")
    print(f"✅ Val: {len(val_dataset)} samples")
    print()
    
    # Initialize model
    print("🏗️  Initializing model...")
    model = StepPredictor(
        input_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT
    )
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    print()
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("🚀 Starting training...")
    print("="*80)
    print()
    
    best_val_mae = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        
        # Validate
        val_loss, val_mae, _, _ = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            
            model_path = Path(Config.MODEL_DIR) / "step_predictor_v1.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'config': {
                    'embedding_dim': Config.EMBEDDING_DIM,
                    'hidden_dims': Config.HIDDEN_DIMS,
                    'dropout': Config.DROPOUT,
                    'min_steps': Config.MIN_STEPS,
                    'max_steps': Config.MAX_STEPS
                }
            }, model_path)
            print(f"   💾 Saved best model (Val MAE: {val_mae:.2f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
        
        print()
    
    print("="*80)
    print("✅ Training complete!")
    print(f"Best validation MAE: {best_val_mae:.2f} steps")
    print()
    
    # Plot training curves
    curves_path = Path(Config.RESULTS_DIR) / "training_curves.png"
    plot_training_curves(train_losses, val_losses, train_maes, val_maes, curves_path)
    print()
    
    # Load best model for evaluation
    print("📊 Evaluating on both thresholds...")
    checkpoint = torch.load(Path(Config.MODEL_DIR) / "step_predictor_v1.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print()
    
    # Evaluate on 0.10 threshold
    df_010 = pd.read_csv(Config.VAL_CSV_010)
    embeddings_010 = extract_clip_embeddings(df_010['prompt'].tolist())
    results_010 = evaluate_predictor(model, df_010, embeddings_010, "0.10", Config.DEVICE)
    
    # Evaluate on 0.12 threshold
    df_012 = pd.read_csv(Config.VAL_CSV_012)
    embeddings_012 = extract_clip_embeddings(df_012['prompt'].tolist())
    results_012 = evaluate_predictor(model, df_012, embeddings_012, "0.12", Config.DEVICE)
    
    # Print results
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print()
    
    for results in [results_010, results_012]:
        print(f"📊 Threshold {results['threshold']}:")
        print("-" * 40)
        print(f"   MAE (continuous): {results['mae']:.2f} steps")
        print(f"   MAE (rounded): {results['mae_rounded']:.2f} steps")
        print(f"   Exact match rate: {results['exact_match']*100:.1f}%")
        print(f"   Within 1 bucket: {results['within_1_step']*100:.1f}%")
        print(f"   Predicted speedup: {results['predicted_speedup']:.1f}%")
        print(f"   Actual speedup: {results['actual_speedup']:.1f}%")
        print()
    
    # Plot predictions
    pred_plot_path = Path(Config.RESULTS_DIR) / "predictions_vs_targets.png"
    plot_predictions_vs_targets(results_010, results_012, pred_plot_path)
    print()
    
    # Save evaluation report
    report_path = Path(Config.RESULTS_DIR) / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("STEP PREDICTOR VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best validation MAE: {best_val_mae:.2f} steps\n\n")
        
        for results in [results_010, results_012]:
            f.write(f"Threshold {results['threshold']}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"MAE (continuous): {results['mae']:.2f} steps\n")
            f.write(f"MAE (rounded): {results['mae_rounded']:.2f} steps\n")
            f.write(f"Exact match rate: {results['exact_match']*100:.1f}%\n")
            f.write(f"Within 1 bucket: {results['within_1_step']*100:.1f}%\n")
            f.write(f"Predicted speedup: {results['predicted_speedup']:.1f}%\n")
            f.write(f"Actual speedup: {results['actual_speedup']:.1f}%\n\n")
    
    print(f"✅ Saved validation report to {report_path}")
    print()
    
    print("="*80)
    print("🎉 ALL DONE!")
    print("="*80)
    print()
    print("Outputs:")
    print(f"   📦 Model: {Path(Config.MODEL_DIR) / 'step_predictor_v1.pth'}")
    print(f"   📊 Training curves: {curves_path}")
    print(f"   📈 Predictions plot: {pred_plot_path}")
    print(f"   📄 Validation report: {report_path}")
    print()

if __name__ == "__main__":
    main()