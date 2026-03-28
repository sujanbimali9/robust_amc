import os
import torch

class Config:
    # Dataset source: 'generated' or 'rml'
    DATA_SOURCE = 'rml' 
    RML_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'RML2016.10a_dict.pkl')
    
    # Dataset parameters
    GENERATED_MODULATIONS = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'PAM4']
    RML_MODULATIONS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
    MODULATIONS = RML_MODULATIONS if DATA_SOURCE == 'rml' else GENERATED_MODULATIONS
    NUM_CLASSES = len(MODULATIONS)
    SAMPLES_PER_SYMBOL = 8
    NUM_SYMBOLS = 128
    SAMPLE_LENGTH = 128 if DATA_SOURCE == 'rml' else (SAMPLES_PER_SYMBOL * NUM_SYMBOLS)
    
    # SNR parameters — boundaries match RadioML 2016.10a range exactly
    SNR_RANGE = (-20, 18)
    SNR_LOW = (-20, 2)     # Widened: overlap helps experts generalize
    SNR_MID = (-2, 12)     # Widened with overlap
    SNR_HIGH = (8, 20)     # Widened with overlap (upper > 18 so snr=18 is included)
    SNR_BINS = ['low', 'mid', 'high']
    NUM_EXPERTS = len(SNR_BINS)
    
    # Channel models
    CHANNELS = ['AWGN', 'Rayleigh', 'Rician']
    
    # Training parameters
    BATCH_SIZE = 512             # Larger batch for more stable gradients
    LEARNING_RATE = 3e-3         # Higher initial LR with cosine annealing
    NUM_EPOCHS = 200             # More epochs for convergence
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 30
    LABEL_SMOOTHING = 0.1        # Helps generalization on noisy signals
    GRAD_CLIP_NORM = 5.0         # Prevent gradient explosions
    
    # Expert-specific training
    EXPERT_LR = 1e-3
    EXPERT_EPOCHS = 150
    EXPERT_PATIENCE = 25
    
    # MoE fine-tuning
    MOE_LR = 5e-4                # End-to-end fine-tuning LR
    MOE_EPOCHS = 200
    MOE_PATIENCE = 30
    
    # SNR estimator training
    SNR_LR = 1e-3
    SNR_EPOCHS = 100
    SNR_PATIENCE = 20
    
    # Model parameters
    SNR_ESTIMATOR_HIDDEN = [128, 64]
    EXPERT_CNN_FILTERS = [64, 128, 256]
    GATING_HIDDEN = [128, 64]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # DataLoader workers (0 for Windows compatibility)
    NUM_WORKERS = 0
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints')
    LOG_PATH = os.path.join(BASE_DIR, 'logs')
    RESULTS_PATH = os.path.join(BASE_DIR, 'results')
    
    @classmethod
    def get_snr_bin(cls, snr):
        """Convert SNR value to bin index"""
        if snr < cls.SNR_LOW[1]:
            return 0  # low
        elif snr < cls.SNR_MID[1]:
            return 1  # mid
        else:
            return 2  # high