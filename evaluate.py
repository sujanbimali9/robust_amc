import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config.config import Config
from data.dataset import IQDataset, load_rml_data
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC, MoEAMCWithAnalysis
from models.expert_networks import ExpertCNN


def load_test_data(config):
    """Load and split data consistently with train.py"""
    np.random.seed(42)  # Same seed as train.py
    
    if config.DATA_SOURCE == 'rml':
        signals, labels, snr_vals, mods = load_rml_data(config.RML_FILE)
    else:
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


def evaluate_model(model, loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        for signals, labels, snrs in tqdm(loader, desc="Testing"):
            signals = signals.to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
            
    return np.array(all_preds), np.array(all_labels), np.array(all_snrs)


def plot_performance(preds, labels, snrs, config, mods):
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    # 1. Accuracy vs SNR
    unique_snrs = sorted(np.unique(snrs))
    snr_accs = []
    for snr in unique_snrs:
        mask = snrs == snr
        acc = 100 * np.mean(preds[mask] == labels[mask])
        snr_accs.append(acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, snr_accs, 'b-o', linewidth=2, markersize=6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Overall Accuracy vs SNR', fontsize=14)
    plt.ylim([0, 105])
    
    # Add text annotation for overall accuracy
    overall_acc = 100 * np.mean(preds == labels)
    plt.axhline(y=overall_acc, color='r', linestyle='--', alpha=0.5, label=f'Overall: {overall_acc:.1f}%')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/accuracy_vs_snr.png", dpi=150)
    plt.close()
    
    # Print per-SNR accuracy table
    print("\nPer-SNR Accuracy:")
    print("-" * 30)
    for snr, acc in zip(unique_snrs, snr_accs):
        print(f"  SNR {snr:+3.0f} dB: {acc:6.2f}%")
    print("-" * 30)
    
    # 2. Confusion Matrix at specific SNRs
    for target_snr in [0, 10, 18]:
        if target_snr in unique_snrs:
            mask = snrs == target_snr
            cm = confusion_matrix(labels[mask], preds[mask], labels=range(len(mods)))
            
            plt.figure(figsize=(12, 10))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=mods, yticklabels=mods, vmin=0, vmax=1)
            plt.title(f'Confusion Matrix at SNR = {target_snr}dB', fontsize=14)
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('True', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{config.RESULTS_PATH}/cm_snr_{target_snr}.png", dpi=150)
            plt.close()
            
            snr_acc = 100 * np.mean(preds[mask] == labels[mask])
            print(f"\nAccuracy at SNR={target_snr}dB: {snr_acc:.2f}%")

    # 3. Overall confusion matrix
    cm = confusion_matrix(labels, preds, labels=range(len(mods)))
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=mods, yticklabels=mods, vmin=0, vmax=1)
    plt.title('Normalized Overall Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/cm_overall.png", dpi=150)
    plt.close()
    
    # 4. Classification report
    report = classification_report(labels, preds, target_names=mods, zero_division=0)
    print(f"\nClassification Report:\n{report}")
    
    # Save report
    with open(f"{config.RESULTS_PATH}/classification_report.txt", 'w') as f:
        f.write(f"Overall Accuracy: {overall_acc:.2f}%\n\n")
        f.write("Per-SNR Accuracy:\n")
        for snr, acc in zip(unique_snrs, snr_accs):
            f.write(f"  SNR {snr:+3.0f} dB: {acc:6.2f}%\n")
        f.write(f"\n{report}")


if __name__ == "__main__":
    config = Config()
    
    print(f"Device: {config.DEVICE}")
    
    test_dataset, mods = load_test_data(config)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)
    
    print(f"Test samples: {len(test_dataset)}")
    
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=len(mods),
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    model_path = f"{config.MODEL_PATH}/moe_amc_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Training might be required.")
        exit(1)
        
    preds, labels, snrs = evaluate_model(model, test_loader, config)
    
    avg_acc = 100 * np.mean(preds == labels)
    print(f"\n{'=' * 40}")
    print(f"  Final Test Accuracy: {avg_acc:.2f}%")
    print(f"{'=' * 40}")
    
    plot_performance(preds, labels, snrs, config, mods)
    print(f"\nResults saved to {config.RESULTS_PATH}")