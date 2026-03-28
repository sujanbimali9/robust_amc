import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import os
import time
from tqdm import tqdm
import json

from config.config import Config
from data.dataset import IQDataset, SNRStratifiedDataset, load_rml_data
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC
from models.expert_networks import ExpertCNN
from models.snr_estimator import SNREstimator


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def __call__(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"  ✓ New best accuracy: {val_acc:.2f}%. Model saved.")
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: creates convex combinations of training examples"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup augmentation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_snr_estimator(train_loader, val_loader, config):
    """Train SNR estimator separately"""
    print("\n" + "=" * 60)
    print(">>> Stage 1: Training SNR Estimator")
    print("=" * 60)
    
    model = SNREstimator(
        input_channels=2,
        hidden_dims=config.SNR_ESTIMATOR_HIDDEN,
        output_dim=config.NUM_EXPERTS
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=config.SNR_LR, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    early_stopping = EarlyStopping(
        patience=config.SNR_PATIENCE, 
        path=f"{config.MODEL_PATH}/snr_estimator_best.pth"
    )
    
    for epoch in range(config.SNR_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"SNR Epoch {epoch+1}/{config.SNR_EPOCHS}", leave=False)
        for signals, labels, snrs in pbar:
            signals = signals.to(config.DEVICE)
            snr_bins = torch.tensor([config.get_snr_bin(s.item()) for s in snrs]).to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, snr_bins)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += snr_bins.size(0)
            correct += predicted.eq(snr_bins).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.1f}%"})
        
        scheduler.step()
        
        # Validation
        val_acc = validate_snr(model, val_loader, config)
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}% | Val SNR Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered for SNR estimator")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/snr_estimator_best.pth"))
    print(f"Best SNR Estimator Accuracy: {early_stopping.best_acc:.2f}%")
    return model


def validate_snr(model, loader, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, _, snrs in loader:
            signals = signals.to(config.DEVICE)
            snr_bins = torch.tensor([config.get_snr_bin(s.item()) for s in snrs]).to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            total += snr_bins.size(0)
            correct += predicted.eq(snr_bins).sum().item()
    return 100. * correct / total


def train_expert(expert_id, train_loader, val_loader, config):
    """Train individual expert for specific SNR range with mixup"""
    print(f"\n{'=' * 60}")
    print(f">>> Stage 2: Training Expert {expert_id} ({config.SNR_BINS[expert_id]} SNR)")
    print(f"    Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"{'=' * 60}")
    
    if len(train_loader.dataset) == 0:
        print(f"Warning: No data for expert {expert_id}. Skipping.")
        return None

    model = ExpertCNN(
        input_channels=2,
        num_classes=config.NUM_CLASSES,
        filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=config.EXPERT_LR,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6)
    
    early_stopping = EarlyStopping(
        patience=config.EXPERT_PATIENCE, 
        path=f"{config.MODEL_PATH}/expert_{expert_id}_best.pth"
    )
    
    for epoch in range(config.EXPERT_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Expert {expert_id} Epoch {epoch+1}", leave=False)
        for signals, labels, _ in pbar:
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # Apply mixup augmentation 50% of the time
            use_mixup = np.random.random() < 0.5
            if use_mixup:
                signals, labels_a, labels_b, lam = mixup_data(signals, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(signals)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            if not use_mixup:
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            else:
                total += labels_a.size(0)
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                          (1 - lam) * predicted.eq(labels_b).sum().item())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        val_acc = validate_expert(model, val_loader, config)
        train_acc = 100. * correct / total if total > 0 else 0
        
        print(f"Expert {expert_id} Epoch {epoch+1}: Train: {train_acc:.1f}% | Val: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"Early stopping for Expert {expert_id}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/expert_{expert_id}_best.pth"))
    print(f"Best Expert {expert_id} Accuracy: {early_stopping.best_acc:.2f}%")
    return model


def validate_expert(model, loader, config):
    model.eval()
    correct = 0
    total = 0
    if len(loader.dataset) == 0: 
        return 0.0
    with torch.no_grad():
        for signals, labels, _ in loader:
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def train_moe_system(train_loader, val_loader, config):
    """Train complete MoE system end-to-end with auxiliary losses"""
    print(f"\n{'=' * 60}")
    print(">>> Stage 3: End-to-End MoE Fine-Tuning")
    print(f"{'=' * 60}")
    
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS,
        gating_mode='soft'
    ).to(config.DEVICE)
    
    # Load pre-trained components
    loaded_components = []
    try:
        snr_path = f"{config.MODEL_PATH}/snr_estimator_best.pth"
        if os.path.exists(snr_path):
            model.snr_estimator.load_state_dict(
                torch.load(snr_path, map_location=config.DEVICE)
            )
            loaded_components.append("SNR Estimator")
        
        for i in range(config.NUM_EXPERTS):
            expert_path = f"{config.MODEL_PATH}/expert_{i}_best.pth"
            if os.path.exists(expert_path):
                model.experts[i].load_state_dict(
                    torch.load(expert_path, map_location=config.DEVICE)
                )
                loaded_components.append(f"Expert {i}")
    except Exception as e:
        print(f"Warning: Could not load all pre-trained weights: {e}")
    
    print(f"  Loaded: {', '.join(loaded_components)}")
    
    # Differential learning rates: lower LR for pre-trained experts, higher for gating
    expert_params = []
    for expert in model.experts:
        expert_params.extend(list(expert.parameters()))
    
    param_groups = [
        {'params': model.snr_estimator.parameters(), 'lr': config.MOE_LR * 0.1},  # Frozen-ish
        {'params': expert_params, 'lr': config.MOE_LR * 0.5},  # Slow updates
        {'params': model.gating.parameters(), 'lr': config.MOE_LR * 2.0},  # Fast gating learning
    ]
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    early_stopping = EarlyStopping(
        patience=config.MOE_PATIENCE, 
        path=f"{config.MODEL_PATH}/moe_amc_best.pth"
    )
    
    # Auxiliary loss weights
    load_balance_weight = 0.1
    diversity_weight = 0.05
    snr_aux_weight = 0.3  # Auxiliary SNR classification loss
    
    snr_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config.MOE_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"MoE Epoch {epoch+1}/{config.MOE_EPOCHS}", leave=False)
        for signals, labels, snrs in pbar:
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            snr_bins = torch.tensor([config.get_snr_bin(s.item()) for s in snrs]).to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward with expert outputs
            final_output, expert_outputs, gating_weights, snr_logits = model(
                signals, return_expert_outputs=True
            )
            
            # Primary classification loss
            cls_loss = criterion(final_output, labels)
            
            # Auxiliary SNR loss (keeps SNR estimator accurate)
            snr_loss = snr_criterion(snr_logits, snr_bins)
            
            # Load balance loss (prevents expert collapse)
            lb_loss = model.get_load_balance_loss(gating_weights)
            
            # Diversity loss (encourages experts to specialize)
            div_loss = model.get_diversity_loss(expert_outputs)
            
            # Total loss
            total_loss = (cls_loss + 
                         snr_aux_weight * snr_loss + 
                         load_balance_weight * lb_loss + 
                         diversity_weight * div_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            train_loss += cls_loss.item()
            _, predicted = final_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'cls': f"{cls_loss.item():.3f}",
                'snr': f"{snr_loss.item():.3f}",
                'acc': f"{100.*correct/total:.1f}%"
            })
        
        scheduler.step()
        
        # Validation
        val_acc = validate_expert(model, val_loader, config)
        train_acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)
        
        print(f"MoE Epoch {epoch+1}: Train: {train_acc:.1f}% | Val: {val_acc:.2f}% | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping for MoE system")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/moe_amc_best.pth", map_location=config.DEVICE))
    print(f"\nBest MoE Accuracy: {early_stopping.best_acc:.2f}%")
    return model


if __name__ == "__main__":
    config = Config()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True
    
    print(f"Device: {config.DEVICE}")
    print(f"Data source: {config.DATA_SOURCE}")
    
    # Data Loading
    if config.DATA_SOURCE == 'rml':
        signals, labels, snr_vals, mods = load_rml_data(config.RML_FILE)
    else:
        generator = SignalGenerator(samples_per_symbol=config.SAMPLES_PER_SYMBOL, num_symbols=config.NUM_SYMBOLS)
        signals, labels, snr_vals = generator.generate_dataset(config.MODULATIONS, config.SNR_RANGE)
    
    # Train/Val/Test split
    n = len(signals)
    indices = np.random.permutation(n)
    train_end = int(config.TRAIN_SPLIT * n)
    val_end = train_end + int(config.VAL_SPLIT * n)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f"\nSplit: {len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test")
    print(f"Classes: {config.NUM_CLASSES}")
    print(f"SNR ranges: Low={config.SNR_LOW}, Mid={config.SNR_MID}, High={config.SNR_HIGH}")
    
    # Create datasets (augmentation ON for training only)
    train_dataset = IQDataset(
        signals[train_indices], labels[train_indices], snr_vals[train_indices],
        normalize=True, augment=True
    )
    val_dataset = IQDataset(
        signals[val_indices], labels[val_indices], snr_vals[val_indices],
        normalize=True, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    start_time = time.time()
    
    # ===== Stage 1: SNR Estimator =====
    train_snr_estimator(train_loader, val_loader, config)
    
    # ===== Stage 2: Expert Networks (with overlapping SNR ranges) =====
    snr_ranges = [config.SNR_LOW, config.SNR_MID, config.SNR_HIGH]
    for i, snr_range in enumerate(snr_ranges):
        expert_train = SNRStratifiedDataset(
            signals[train_indices], labels[train_indices], snr_vals[train_indices], 
            snr_range, normalize=True, augment=True
        )
        expert_val = SNRStratifiedDataset(
            signals[val_indices], labels[val_indices], snr_vals[val_indices], 
            snr_range, normalize=True, augment=False
        )
        
        e_train_loader = DataLoader(
            expert_train, batch_size=config.BATCH_SIZE, shuffle=True, 
            num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=len(expert_train) > config.BATCH_SIZE
        )
        e_val_loader = DataLoader(
            expert_val, batch_size=config.BATCH_SIZE, shuffle=False, 
            num_workers=config.NUM_WORKERS, pin_memory=True
        )
        
        train_expert(i, e_train_loader, e_val_loader, config)
    
    # ===== Stage 3: End-to-End MoE Fine-tuning =====
    model = train_moe_system(train_loader, val_loader, config)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed/60:.1f} minutes")
    print(f"Checkpoints saved to '{config.MODEL_PATH}'")
    print(f"{'=' * 60}")