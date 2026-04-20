import csv
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from neural_transceiver import CaduceusAE


CHANNELS = ("awgn", "rayleigh", "rician")
SNR_DB_VALUES = list(range(0, 22, 2))
SYMBOLS_PER_POINT = 10_000


def apply_channel_complex(symbols, noise_std, channel_type):
    device = symbols.device
    dtype = symbols.real.dtype

    if channel_type == "awgn":
        faded = symbols
    elif channel_type == "rayleigh":
        h = torch.complex(
            torch.randn_like(symbols.real, dtype=dtype, device=device),
            torch.randn_like(symbols.real, dtype=dtype, device=device),
        ) / math.sqrt(2.0)
        faded = h * symbols
    elif channel_type == "rician":
        k_factor = 3.0
        los_scale = math.sqrt(k_factor / (k_factor + 1.0))
        scatter_scale = math.sqrt(1.0 / (k_factor + 1.0))
        scatter = torch.complex(
            torch.randn_like(symbols.real, dtype=dtype, device=device),
            torch.randn_like(symbols.real, dtype=dtype, device=device),
        ) / math.sqrt(2.0)
        h = los_scale + scatter_scale * scatter
        faded = h * symbols
    else:
        raise ValueError(f"Unsupported channel type: {channel_type}")

    noise = torch.complex(
        torch.randn_like(symbols.real, dtype=dtype, device=device),
        torch.randn_like(symbols.real, dtype=dtype, device=device),
    ) * noise_std
    return faded + noise


def ber_from_bits(tx_bits, rx_bits):
    return (tx_bits != rx_bits).float().mean().item()


def evaluate_qpsk(noise_std, channel_type, symbols_count, device):
    tx_bits = torch.randint(0, 2, (symbols_count, 2), device=device, dtype=torch.int64)
    i = tx_bits[:, 0].float() * 2 - 1
    q = tx_bits[:, 1].float() * 2 - 1
    tx_symbols = torch.complex(i, q) / math.sqrt(2.0)
    rx_symbols = apply_channel_complex(tx_symbols, noise_std, channel_type)

    rx_bits = torch.zeros_like(tx_bits)
    rx_bits[:, 0] = (rx_symbols.real > 0).long()
    rx_bits[:, 1] = (rx_symbols.imag > 0).long()
    return ber_from_bits(tx_bits, rx_bits)


def evaluate_16qam(noise_std, channel_type, symbols_count, device):
    tx_bits = torch.randint(0, 2, (symbols_count, 4), device=device, dtype=torch.int64)

    # Gray-coded mapping for each axis: 00->-3, 01->-1, 11->+1, 10->+3
    def bits_to_level(b0, b1):
        return torch.where(
            (b0 == 0) & (b1 == 0),
            torch.full_like(b0, -3.0, dtype=torch.float32),
            torch.where(
                (b0 == 0) & (b1 == 1),
                torch.full_like(b0, -1.0, dtype=torch.float32),
                torch.where(
                    (b0 == 1) & (b1 == 1),
                    torch.full_like(b0, 1.0, dtype=torch.float32),
                    torch.full_like(b0, 3.0, dtype=torch.float32),
                ),
            ),
        )

    i = bits_to_level(tx_bits[:, 0], tx_bits[:, 1]).to(device)
    q = bits_to_level(tx_bits[:, 2], tx_bits[:, 3]).to(device)
    tx_symbols = torch.complex(i, q) / math.sqrt(10.0)  # Normalize average power to 1
    rx_symbols = apply_channel_complex(tx_symbols, noise_std, channel_type)

    scaled_i = rx_symbols.real * math.sqrt(10.0)
    scaled_q = rx_symbols.imag * math.sqrt(10.0)
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device)

    def quantize_axis(axis_values):
        idx = torch.argmin(torch.abs(axis_values.unsqueeze(1) - levels.unsqueeze(0)), dim=1)
        return levels[idx]

    qi = quantize_axis(scaled_i)
    qq = quantize_axis(scaled_q)

    def level_to_bits(level):
        b0 = torch.where(level > 0, 1, 0)
        b1 = torch.where(torch.abs(level) < 2, 1, 0)
        return b0.long(), b1.long()

    b0, b1 = level_to_bits(qi)
    b2, b3 = level_to_bits(qq)
    rx_bits = torch.stack((b0, b1, b2, b3), dim=1)
    return ber_from_bits(tx_bits, rx_bits)


def evaluate_caduceus(model, noise_std, channel_type, symbols_count, device):
    labels = torch.randint(0, 16, (symbols_count,), device=device)
    inputs = F.one_hot(labels, num_classes=16).float()

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits, _ = model(inputs, noise_std, channel_type=channel_type)

    predictions = torch.argmax(logits.float(), dim=1)
    tx_bits = ((labels.unsqueeze(1) >> torch.arange(4, device=device)) & 1).long()
    rx_bits = ((predictions.unsqueeze(1) >> torch.arange(4, device=device)) & 1).long()
    return ber_from_bits(tx_bits, rx_bits)


def run_multi_channel_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}")
    os.makedirs("results/ber_curves", exist_ok=True)

    model = CaduceusAE().to(device)
    model.load_state_dict(torch.load("caduceus_model.pth", map_location=device, weights_only=True))
    model.eval()

    rows = []
    for channel in CHANNELS:
        for snr_db in SNR_DB_VALUES:
            noise_std = 10 ** (-snr_db / 20.0)
            with torch.no_grad():
                cad_ber = evaluate_caduceus(model, noise_std, channel, SYMBOLS_PER_POINT, device)
                qpsk_ber = evaluate_qpsk(noise_std, channel, SYMBOLS_PER_POINT, device)
                qam16_ber = evaluate_16qam(noise_std, channel, SYMBOLS_PER_POINT, device)

            rows.append(
                {
                    "channel": channel,
                    "snr_db": snr_db,
                    "caduceus_ber": cad_ber,
                    "qpsk_ber": qpsk_ber,
                    "qam16_ber": qam16_ber,
                }
            )
            print(
                f"[{channel}] SNR={snr_db:2d}dB | "
                f"CADUCEUS={cad_ber:.6f} | QPSK={qpsk_ber:.6f} | 16-QAM={qam16_ber:.6f}"
            )

    csv_path = "results/ber_curves/multi_channel_ber.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["channel", "snr_db", "caduceus_ber", "qpsk_ber", "qam16_ber"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, channel in enumerate(CHANNELS):
        channel_rows = [r for r in rows if r["channel"] == channel]
        snrs = [r["snr_db"] for r in channel_rows]
        axes[idx].semilogy(snrs, [r["caduceus_ber"] for r in channel_rows], "o-", label="CADUCEUS")
        axes[idx].semilogy(snrs, [r["qpsk_ber"] for r in channel_rows], "s-", label="QPSK")
        axes[idx].semilogy(snrs, [r["qam16_ber"] for r in channel_rows], "^-", label="16-QAM")
        axes[idx].set_title(channel.upper())
        axes[idx].set_xlabel("SNR (dB)")
        axes[idx].grid(True, which="both", alpha=0.3)
        if idx == 0:
            axes[idx].set_ylabel("BER")
            axes[idx].legend()

    plt.tight_layout()
    plot_path = "results/ber_curves/multi_channel_ber.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    run_multi_channel_benchmark()
