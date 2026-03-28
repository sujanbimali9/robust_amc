import argparse
import time
import numpy as np
import torch
import scipy.signal as signal_pkg
import threading
import queue

try:
    from rtlsdr import RtlSdr
except ImportError:
    print("Error: The 'pyrtlsdr' package is not installed.")
    print("Please install it using: pip install pyrtlsdr")
    exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("Error: The 'sounddevice' package is not installed.")
    print("Please install it using: pip install sounddevice")
    exit(1)

from config.config import Config
from models.moe_amc import MoEAMC

def demodulate_and_audio_format(signal, modulation, rf_rate, audio_rate=48000):
    """
    Demodulates signal and returns it properly resampled and scaled for audio playback.
    """
    # Calculate exact rational resampling factors for direct RF -> Audio
    # e.g., 48000 / 2048000 = 3 / 128
    from math import gcd
    g = gcd(int(audio_rate), int(rf_rate))
    up = int(audio_rate // g)
    down = int(rf_rate // g)
    
    if modulation in ['AM-DSB', 'AM-SSB']:
        # Envelope detection
        audio = np.abs(signal)
        audio = audio - np.mean(audio)  # Remove DC
        audio = signal_pkg.resample_poly(audio, up, down)
        
    elif modulation in ['WBFM', 'CPFSK', 'GFSK']:
        # FM requires decimation first to prevent noise and unwrap slowness
        # Decimate from 2048000 to 256000 (factor of 8)
        dec_rf = 8
        sig_dec = signal_pkg.resample_poly(signal, 1, dec_rf)
        new_rate = rf_rate / dec_rf
        
        # Discriminator (differentiate phase)
        phase = np.unwrap(np.angle(sig_dec))
        audio = np.diff(phase)
        # Pad back 1 sample lost during diff to keep exact lengths
        audio = np.append(audio, audio[-1])
        
        # Resample to final audio rate
        g_fm = gcd(int(audio_rate), int(new_rate))
        up_fm = int(audio_rate // g_fm)
        down_fm = int(new_rate // g_fm)
        audio = signal_pkg.resample_poly(audio, up_fm, down_fm)
             
    else:
        # Digital modes or undefined: map real part as noise tone
        audio = signal_pkg.resample_poly(np.real(signal), up, down)

    # Normalize audio to float32 range [-1.0, 1.0]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)

def classify_windows(samples, model, config):
    samples_per_window = config.SAMPLE_LENGTH
    num_windows = len(samples) // samples_per_window
    
    if num_windows == 0:
        return config.MODULATIONS[0]  # Fallback
        
    # Evaluate up to 500 windows randomly to save memory on large durations
    max_eval_windows = min(num_windows, 500)
    indices = np.random.choice(num_windows, max_eval_windows, replace=False)
    
    processed_windows = []
    for idx in indices:
        window = samples[idx * samples_per_window : (idx + 1) * samples_per_window]
        power = np.mean(np.abs(window)**2)
        if power > 0:
            window_norm = window / np.sqrt(power)
        else:
            window_norm = window
            
        window_tensor = np.stack([window_norm.real, window_norm.imag], axis=0)
        processed_windows.append(window_tensor)
        
    batch_tensor = torch.FloatTensor(np.array(processed_windows)).to(config.DEVICE)
    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predictions = outputs.max(1)
        
    preds = predictions.cpu().numpy()
    unique, counts = np.unique(preds, return_counts=True)
    majority_class_idx = unique[np.argmax(counts)]
    return config.MODULATIONS[majority_class_idx]

def record_classify_play(sdr, model, config, duration, sample_rate):
    print(f"\nRecording {duration} seconds of data...")
    total_samples = int(duration * sample_rate)
    
    chunk_size = 1024 * 1024
    num_chunks = total_samples // chunk_size
    remainder = total_samples % chunk_size
    
    samples_list = []
    for _ in range(num_chunks):
        samples_list.append(sdr.read_samples(chunk_size))
    if remainder > 0:
        samples_list.append(sdr.read_samples(remainder))
        
    samples = np.concatenate(samples_list)
    print("Finished recording. Running classification...")
    
    mod_type = classify_windows(samples, model, config)
    print(f"Detected Modulation: {mod_type}")
    
    print("Demodulating signal...")
    audio = demodulate_and_audio_format(samples, mod_type, sample_rate, audio_rate=48000)
    
    print("Playing audio...")
    sd.play(audio, samplerate=48000)
    sd.wait()
    print("Playback finished.")

class AudioBuffer:
    def __init__(self, sample_rate):
        self.lock = threading.Lock()
        self.data = np.zeros(0, dtype=np.float32)
        self.sample_rate = sample_rate
        
    def write(self, samples):
        with self.lock:
            self.data = np.concatenate((self.data, samples))
            # Just keep it from growing infinity, limit to 2 seconds
            if len(self.data) > self.sample_rate * 2:
                # hard sync if severely backed up
                self.data = self.data[-self.sample_rate:]
                
    def read(self, frames):
        with self.lock:
            if len(self.data) >= frames:
                out = self.data[:frames]
                self.data = self.data[frames:]
                return out
            else:
                out = np.zeros(frames, dtype=np.float32)
                if len(self.data) > 0:
                    out[:len(self.data)] = self.data
                self.data = np.zeros(0, dtype=np.float32)
                return out

def stream_realtime_audio(sdr, model, config, args):
    print("Starting continuous live classification with realtime audio (Press Ctrl+C to stop)...")
    print("=" * 60)
    
    stream_audio_rate = 48000
    chunk_size = 1024 * 256  # 125ms of RF data
    audio_chunk_len = 6144   # exact audio samples for 125ms (1024 * 256 * 3 / 128 = 6144)
    
    state = {
        "mod_type": config.MODULATIONS[0],
        "latest_samples": (None, 0),
        "running": True,
        "last_print_time": 0
    }
    
    # 1. AI Classifier Thread
    def classifier_worker():
        last_processed_id = -1
        while state["running"]:
            samples, sample_id = state["latest_samples"]
            now = time.time()
            if samples is not None and sample_id != last_processed_id:
                if (now - state["last_print_time"] >= args.interval):
                    new_mod = classify_windows(samples, model, config)
                    last_processed_id = sample_id
                    
                    print(f"[{time.strftime('%H:%M:%S')}] Detected Modulation: {new_mod}")
                    state["last_print_time"] = time.time()
                    state["mod_type"] = new_mod
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    clf_thread = threading.Thread(target=classifier_worker, daemon=True)
    clf_thread.start()
    
    # 2. Dedicated SDR Reader Thread
    rf_q = queue.Queue(maxsize=10)
    def sdr_worker():
        while state["running"]:
            try:
                samples = sdr.read_samples(chunk_size)
                rf_q.put(samples)
            except Exception as e:
                print(f"SDR Read Error: {e}")
                state["running"] = False
                break
                
    sdr_read_thread = threading.Thread(target=sdr_worker, daemon=True)
    sdr_read_thread.start()
    
    # 3. Audio Callback Layer
    audio_buf = AudioBuffer(stream_audio_rate)
    def audio_callback(outdata, frames, time_info, status):
        chunk = audio_buf.read(frames)
        outdata[:, 0] = chunk

    # 4. Demodulation Compute Loop
    sample_id_counter = 0
    
    # Rolling history of 3 blocks: [past, present, future] to fully isolate FIR filter edge ringing
    rf_history = np.zeros(chunk_size * 3, dtype=np.complex64)
    
    try:
        with sd.OutputStream(samplerate=stream_audio_rate, channels=1, callback=audio_callback):
            while state["running"]:
                try:
                    samples = rf_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                sample_id_counter += 1
                state["latest_samples"] = (samples, sample_id_counter)
                
                # Shift buffer sequence: drop oldest, insert newest 
                rf_history = np.concatenate((rf_history[chunk_size:], samples))
                    
                # Process the entire 3-block pipeline
                audio = demodulate_and_audio_format(rf_history, state["mod_type"], sdr.sample_rate, stream_audio_rate)
                
                # Extract ONLY the middle block which is 100% shielded from filter edge transients
                audio = audio[audio_chunk_len : audio_chunk_len * 2]
                
                # Push flawless insulated audio to ringbuffer
                audio_buf.write(audio)
    finally:
        state["running"] = False

def main():
    parser = argparse.ArgumentParser(description="Live RTL-SDR Modulation Classification and Demodulation")
    parser.add_argument("--freq", type=float, required=True, help="Center frequency in Hz (e.g., 100e6 for 100 MHz)")
    parser.add_argument("--sample-rate", type=float, default=2.048e6, help="Sample rate in Hz (default: 2.048e6)")
    parser.add_argument("--gain", type=str, default="auto", help="SDR receiver gain in dB, or 'auto'")
    parser.add_argument("--interval", type=float, default=3.0, help="Interval between continuous classification runs in seconds")
    parser.add_argument("--duration", type=float, default=0.0, help="If > 0, records for this many seconds, evaluates, and plays the audio automatically. If 0, runs continuously without audio playback.")
    args = parser.parse_args()

    config = Config()
    
    print(f"Initializing MoEAMC Model on {config.DEVICE}...")
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=len(config.MODULATIONS),
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    model_path = f"{config.MODEL_PATH}/moe_amc_best.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        print(f"Loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {model_path}.")
        exit(1)

    print("\nConnecting to RTL-SDR...")
    try:
        sdr = RtlSdr()
    except Exception as e:
        print(f"Failed to initialize RTL-SDR: {e}")
        exit(1)

    sdr.sample_rate = args.sample_rate
    sdr.center_freq = args.freq
    
    if args.gain.lower() == 'auto':
        sdr.gain = 'auto'
    else:
        try:
            sdr.gain = float(args.gain)
        except ValueError:
            sdr.gain = 'auto'

    print(f"SDR Configured | Freq: {sdr.center_freq/1e6:.2f} MHz | SR: {sdr.sample_rate/1e6:.2f} MHz | Gain: {sdr.gain}")
    
    try:
        if args.duration > 0.0:
            record_classify_play(sdr, model, config, args.duration, sdr.sample_rate)
        else:
            stream_realtime_audio(sdr, model, config, args)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sdr.close()
        print("SDR device closed.")

if __name__ == "__main__":
    main()
