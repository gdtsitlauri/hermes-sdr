import csv
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from neural_transceiver import CaduceusAE


def estimate_mutual_information(constellation, noise_std, samples_per_symbol=250):
    """
    Monte Carlo estimate of I(X;Y) for equiprobable CADUCEUS symbols on AWGN.
    """
    device = constellation.device
    m = constellation.shape[0]

    x = constellation.unsqueeze(1).repeat(1, samples_per_symbol, 1).reshape(-1, 2)
    symbol_indices = torch.arange(m, device=device).unsqueeze(1).repeat(1, samples_per_symbol).reshape(-1)

    noise = torch.randn_like(x) * noise_std
    y = x + noise

    diff = y.unsqueeze(1) - constellation.unsqueeze(0)  # [N, M, 2]
    dist2 = torch.sum(diff * diff, dim=2)

    sigma2 = noise_std * noise_std
    log_py_given_x = -dist2[torch.arange(dist2.shape[0], device=device), symbol_indices] / (2.0 * sigma2)
    log_py = torch.logsumexp(-dist2 / (2.0 * sigma2), dim=1) - math.log(m)

    mi_nats = (log_py_given_x - log_py).mean()
    mi_bits = (mi_nats / math.log(2.0)).item()
    return max(0.0, min(mi_bits, math.log2(m)))


def run_information_theory_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/theory", exist_ok=True)

    model = CaduceusAE().to(device)
    model.load_state_dict(torch.load("caduceus_model.pth", map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        labels = torch.arange(0, 16, device=device)
        one_hot = F.one_hot(labels, num_classes=16).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, constellation = model(one_hot, noise_std=1e-4, channel_type="awgn")
        constellation = constellation.float()

    snr_db_values = list(range(0, 22, 2))
    rows = []
    for snr_db in snr_db_values:
        snr_linear = 10 ** (snr_db / 10.0)
        shannon_capacity = math.log2(1.0 + snr_linear)  # B=1 Hz normalized
        noise_std = 10 ** (-snr_db / 20.0)
        mi = estimate_mutual_information(constellation, noise_std=noise_std, samples_per_symbol=250)
        rows.append(
            {
                "snr_db": snr_db,
                "shannon_capacity_bphz": shannon_capacity,
                "caduceus_mutual_information": mi,
                "caduceus_spectral_efficiency": mi,
                "qpsk_spectral_efficiency": 2.0,
                "qam16_spectral_efficiency": 4.0,
            }
        )
        print(f"SNR={snr_db:2d} dB | Shannon={shannon_capacity:.4f} | CADUCEUS I(X;Y)={mi:.4f}")

    csv_path = "results/theory/shannon_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "snr_db",
                "shannon_capacity_bphz",
                "caduceus_mutual_information",
                "caduceus_spectral_efficiency",
                "qpsk_spectral_efficiency",
                "qam16_spectral_efficiency",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    snrs = [row["snr_db"] for row in rows]
    shannon = [row["shannon_capacity_bphz"] for row in rows]
    caduceus_mi = [row["caduceus_mutual_information"] for row in rows]

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    plt.plot(snrs, shannon, "k-", label="Shannon Capacity (B=1 Hz)")
    plt.plot(snrs, caduceus_mi, "r-o", label="CADUCEUS Mutual Information")
    plt.ylabel("Bits/s/Hz")
    plt.title("Information-Theoretic Analysis: CADUCEUS vs Shannon Limit")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(snrs, caduceus_mi, "r-o", label="CADUCEUS")
    plt.plot(snrs, [2.0] * len(snrs), "b--", label="QPSK")
    plt.plot(snrs, [4.0] * len(snrs), "g--", label="16-QAM")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Spectral Efficiency (bits/symbol)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_path = "results/theory/shannon_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    run_information_theory_analysis()
