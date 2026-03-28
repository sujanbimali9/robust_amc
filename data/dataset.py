import torch
from torch.utils.data import Dataset
import numpy as np
import pickle


class IQDataset(Dataset):
    def __init__(self, signals, labels, snr_values, normalize=True, augment=False):
        """
        Args:
            signals: Complex I/Q samples (N, L) where L is sample length
            labels: Modulation class labels (N,)
            snr_values: SNR values in dB (N,)
            normalize: Whether to normalize each sample to unit power
            augment: Whether to apply data augmentation
        """
        self.signals = signals
        self.labels = labels
        self.snr_values = snr_values
        self.normalize = normalize
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()  # copy to avoid mutating original

        if self.augment:
            signal = self._augment(signal)

        if self.normalize:
            # Normalize to unit power: x / sqrt(E[|x|^2])
            power = np.mean(np.abs(signal) ** 2)
            if power > 0:
                signal = signal / np.sqrt(power)

        # Convert complex signal to 2-channel real representation (2, L)
        signal_tensor = np.stack([signal.real, signal.imag], axis=0)
        signal_tensor = torch.FloatTensor(signal_tensor)

        label = torch.LongTensor([self.labels[idx]])[0]
        snr = torch.FloatTensor([self.snr_values[idx]])[0]

        return signal_tensor, label, snr

    def _augment(self, signal):
        """Apply I/Q data augmentation — conservative to preserve modulation info"""
        # 1. Random small phase rotation (safe for all modulations)
        if np.random.random() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            signal = signal * np.exp(1j * angle)

        # 2. Random circular time shift (small shift only)
        if np.random.random() < 0.3:
            max_shift = max(1, len(signal) // 8)
            shift = np.random.randint(-max_shift, max_shift)
            signal = np.roll(signal, shift)

        # 3. Small additive Gaussian noise (SNR-aware augmentation)
        if np.random.random() < 0.2:
            noise_power = np.mean(np.abs(signal) ** 2) * 0.01  # -20dB noise
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
            )
            signal = signal + noise

        # 4. Random amplitude scaling (very small range)
        if np.random.random() < 0.2:
            scale = np.random.uniform(0.9, 1.1)
            signal = signal * scale

        return signal


def load_rml_data(file_path):
    """
    Load RML2016.10a dataset from pickle file
    """
    print(f"Loading dataset from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        # Try different encoding if latin1 fails
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

    # Get sorted lists of modulations and SNRs
    mods, snrs = map(lambda x: sorted(list(set(x))), zip(*data.keys()))

    X = []
    lbl = []
    snr_vals = []

    for mod in mods:
        for snr in snrs:
            X_item = data[(mod, snr)]
            # X_item is (1000, 2, 128)
            # Convert to complex (1000, 128)
            X_complex = X_item[:, 0, :] + 1j * X_item[:, 1, :]
            X.append(X_complex)

            for _ in range(X_item.shape[0]):
                lbl.append(mod)
                snr_vals.append(snr)

    X = np.vstack(X)

    # Map modulation names to indices
    mod_to_idx = {mod: i for i, mod in enumerate(mods)}
    lbl = np.array([mod_to_idx[l] for l in lbl])
    snr_vals = np.array(snr_vals)

    print(f"Dataset loaded: {X.shape[0]} samples, {len(mods)} classes")
    print(f"  Modulations: {mods}")
    print(f"  SNR range: [{min(snr_vals)}, {max(snr_vals)}] dB")
    return X, lbl, snr_vals, mods


class SNRStratifiedDataset(IQDataset):
    def __init__(self, signals, labels, snr_values, snr_range, normalize=True, augment=False):
        """
        Dataset filtered by SNR range (inclusive on both ends)
        """
        mask = (snr_values >= snr_range[0]) & (snr_values <= snr_range[1])
        super().__init__(signals[mask], labels[mask], snr_values[mask],
                         normalize=normalize, augment=augment)