import torch
import numpy as np
import matplotlib.pyplot as plt

def run_spectrum_analysis():
    # Χρήση της GTX 1650 (cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HERMES Analyzer running on: {device}")

    # Δημιουργία σύνθετου σήματος (π.χ. 3 φορείς με θόρυβο)
    fs = 1000  # Συχνότητα δειγματοληψίας
    t = torch.linspace(0, 1, fs, device=device)
    # Σήμα = 50Hz + 120Hz + Λευκός Θόρυβος
    signal = torch.sin(2 * np.pi * 50 * t) + 0.5 * torch.sin(2 * np.pi * 120 * t)
    signal += torch.randn_like(signal) * 0.2

    # GPU-Accelerated FFT
    fft_result = torch.fft.fft(signal)
    magnitudes = torch.abs(fft_result).cpu().numpy()
    freqs = np.fft.fftfreq(fs, d=1/fs)

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:fs//2], magnitudes[:fs//2])
    plt.title("HERMES - Real-time Spectrum Analysis (GPU Accelerated)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_spectrum_analysis()