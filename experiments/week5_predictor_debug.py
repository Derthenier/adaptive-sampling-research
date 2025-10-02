import torch
import pandas as pd
import numpy as np

# Load the model and data
checkpoint = torch.load('models/step_predictor_v1.pth', weights_only=False)
df = pd.read_csv('data/training_data_012.csv')

# Load embeddings
cache = torch.load('data/clip_embeddings.pt')
embeddings = cache['embeddings']

# Load model
from week5_train_step_predictor import StepPredictor
model = StepPredictor()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
predictions = []
with torch.no_grad():
    for emb in embeddings:
        pred = model(emb.unsqueeze(0))
        predictions.append(pred.item())

predictions = np.array(predictions)
targets = df['optimal_steps'].values

# Analyze predictions
print("="*60)
print("PREDICTION ANALYSIS")
print("="*60)
print(f"Predicted steps - Min: {predictions.min():.1f}, Max: {predictions.max():.1f}, Mean: {predictions.mean():.1f}")
print(f"Actual steps - Min: {targets.min():.1f}, Max: {targets.max():.1f}, Mean: {targets.mean():.1f}")
print()

# Are predictions all similar? (collapsed to mean?)
print(f"Prediction std: {predictions.std():.2f}")
print(f"Actual std: {targets.std():.2f}")
print()

# Show some examples
print("Sample predictions vs targets:")
for i in range(min(10, len(df))):
    print(f"  '{df.iloc[i]['prompt'][:50]}...'")
    print(f"    Predicted: {predictions[i]:.1f}, Actual: {targets[i]}")
print()

# How many predict < 25? (aggressive)
aggressive = (predictions < 25).sum()
print(f"Predictions < 25 steps: {aggressive}/{len(predictions)} ({aggressive/len(predictions)*100:.1f}%)")
print(f"Actuals < 25 steps: {(targets < 25).sum()}/{len(targets)} ({(targets < 25).sum()/len(targets)*100:.1f}%)")