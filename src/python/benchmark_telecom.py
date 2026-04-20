import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from neural_transceiver import CaduceusAE

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaduceusAE().to(device)
    
    os.makedirs("results/ber_curves", exist_ok=True)

    try:
        model.load_state_dict(torch.load("caduceus_model.pth"))
        model.eval()
        print("Loaded trained CADUCEUS-WAVE model.")
    except:
        print("Error: caduceus_model.pth not found!")
        return

    snr_range = np.linspace(0, 20, 15)
    ber_results = []

    for snr in snr_range:
        noise_std = 10**(-snr/20)
        total = 50000
        with torch.no_grad():
            labels = torch.randint(0, 16, (total,), device=device)
            inputs = torch.nn.functional.one_hot(labels, num_classes=16).float()
            outputs, _ = model(inputs, noise_std)
            predictions = torch.argmax(outputs, dim=1)
            errors = (predictions != labels).sum().item()
        
        ber_results.append(errors / total)
        print(f"SNR: {snr:.1f}dB | BER: {ber_results[-1]:.5f}")

    plt.figure(figsize=(10,6))
    plt.semilogy(snr_range, ber_results, 'b-o', linewidth=2, label='CADUCEUS-WAVE')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('HERMES Performance Benchmark')
    plt.grid(True, which='both')
    plt.legend()
    
    # Save before showing
    plt.savefig("results/ber_curves/ber_curve_results.png")
    print("Plot saved to results/ber_curves/ber_curve_results.png")
    plt.show()

if __name__ == "__main__":
    run_benchmark()