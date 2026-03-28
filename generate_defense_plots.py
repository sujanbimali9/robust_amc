import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from config.config import Config
from data.dataset import IQDataset, load_rml_data
from models.moe_amc import MoEAMC

# Use non-interactive backend for headless environments
plt.switch_backend('agg')

def load_test_data(config):
    """Load and split data consistently with train.py"""
    np.random.seed(42)
    
    if config.DATA_SOURCE == 'rml':
        signals, labels, snr_vals, mods = load_rml_data(config.RML_FILE)
    else:
        from data.generator import SignalGenerator
        generator = SignalGenerator(samples_per_symbol=config.SAMPLES_PER_SYMBOL, num_symbols=config.NUM_SYMBOLS)
        signals, labels, snr_vals = generator.generate_dataset(config.MODULATIONS, config.SNR_RANGE)
        mods = config.MODULATIONS
    
    n = len(signals)
    indices = np.random.permutation(n)
    train_end = int(config.TRAIN_SPLIT * n)
    val_end = train_end + int(config.VAL_SPLIT * n)
    
    test_indices = indices[val_end:]
    
    test_dataset = IQDataset(signals[test_indices], labels[test_indices], snr_vals[test_indices],
                             normalize=True, augment=False)
    return test_dataset, mods

def run_inference(model, loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    all_snrs = []
    all_weights = []
    
    with torch.no_grad():
        for signals, labels, snrs in tqdm(loader, desc="Gathering results"):
            signals = signals.to(config.DEVICE)
            # Forward with expert analysis
            final_output, _, gating_weights, _ = model(signals, return_expert_outputs=True)
            _, predicted = final_output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
            all_weights.extend(gating_weights.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels), np.array(all_snrs), np.array(all_weights)

def generate_plots(preds, labels, snrs, weights, config, mods):
    print("\nGenerating professional plots for defense...")
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- 1. Accuracy vs SNR ---
    unique_snrs = sorted(np.unique(snrs))
    snr_accs = [100 * np.mean(preds[snrs == snr] == labels[snrs == snr]) for snr in unique_snrs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, snr_accs, 'b-o', linewidth=2.5, markersize=8, label='MoE AMC')
    plt.axhline(y=100 * np.mean(preds == labels), color='r', linestyle='--', alpha=0.6, 
                label=f'Avg Acc: {100*np.mean(preds==labels):.1f}%')
    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title('Overall Accuracy vs. Signal-to-Noise Ratio (SNR)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/defense_accuracy_vs_snr.png", dpi=300)
    plt.close()

    # --- 2. Expert Gating Map (Heatmap of expert weights vs SNR) ---
    # This shows how the model switches experts as SNR increases
    expert_snr_weights = []
    for snr in unique_snrs:
        mask = snrs == snr
        avg_w = weights[mask].mean(axis=0)
        expert_snr_weights.append(avg_w)
    expert_snr_weights = np.array(expert_snr_weights).T # (num_experts, num_snrs)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(expert_snr_weights, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=[f"{int(s)}" for s in unique_snrs],
                yticklabels=[f"Expert {config.SNR_BINS[i]}" for i in range(len(config.SNR_BINS))])
    plt.title('Expert Gating Activation Map Across SNR Regimes', fontsize=15, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('Expert Module', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/defense_expert_gating_map.png", dpi=300)
    plt.close()

    # --- 3. Normalized Overall Confusion Matrix ---
    cm = confusion_matrix(labels, preds, labels=range(len(mods)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=mods, yticklabels=mods, vmin=0, vmax=1)
    plt.title('Normalized Overall Confusion Matrix (All SNRs)', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted Modulation', fontsize=13)
    plt.ylabel('True Modulation', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/defense_confusion_matrix_overall.png", dpi=300)
    plt.close()

    # --- 4. Confusion Matrix at High SNR (+18 dB) ---
    target_snr = 18
    if target_snr in unique_snrs:
        mask = snrs == target_snr
        cm_high = confusion_matrix(labels[mask], preds[mask], labels=range(len(mods)))
        cm_high_norm = cm_high.astype('float') / cm_high.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(13, 11))
        sns.heatmap(cm_high_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=mods, yticklabels=mods, vmin=0, vmax=1)
        plt.title(f'Classification Performance at High Signal Quality (SNR = {target_snr}dB)', 
                  fontsize=15, fontweight='bold')
        plt.xlabel('Predicted Modulation', fontsize=13)
        plt.ylabel('True Modulation', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_PATH}/defense_confusion_matrix_snr_18.png", dpi=300)
        plt.close()

    # --- 5. Per-Class Accuracy vs SNR (Top 5 Classes) ---
    top_n = 6
    per_class_snr_acc = {}
    for i, mod in enumerate(mods):
        accs = []
        for snr in unique_snrs:
            mask = (snrs == snr) & (labels == i)
            if np.any(mask):
                accs.append(100 * np.mean(preds[mask] == labels[mask]))
            else:
                accs.append(None)
        per_class_snr_acc[mod] = accs

    plt.figure(figsize=(12, 7))
    # Plot top 6 most accurate (avg) classes
    avg_per_class = {mod: np.nanmean([a for a in accs if a is not None]) for mod, accs in per_class_snr_acc.items()}
    sorted_mods = sorted(avg_per_class.keys(), key=lambda x: avg_per_class[x], reverse=True)[:top_n]
    
    for mod in sorted_mods:
        valid = [i for i, a in enumerate(per_class_snr_acc[mod]) if a is not None]
        plt.plot(np.array(unique_snrs)[valid], np.array(per_class_snr_acc[mod])[valid], 
                 marker='s', linewidth=2, markersize=5, label=mod)
        
    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title(f'Performance Comparison: Top {top_n} Modulation Types', fontsize=15, fontweight='bold')
    plt.legend(title="Modulation Type", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/defense_top_mods_accuracy.png", dpi=300)
    plt.close()

    print(f"All plots saved to: {config.RESULTS_PATH}")

if __name__ == "__main__":
    config = Config()
    
    test_dataset, mods = load_test_data(config)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                             num_workers=config.NUM_WORKERS)
    
    print(f"Evaluating MoE model for defense plotting...")
    
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=len(mods),
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    model_path = f"{config.MODEL_PATH}/moe_amc_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print(f"Successfully loaded best matching model from {model_path}")
    else:
        print(f"!!! Error: best model not found at {model_path}")
        exit(1)
        
    preds, labels, snrs, weights = run_inference(model, test_loader, config)
    generate_plots(preds, labels, snrs, weights, config, mods)
