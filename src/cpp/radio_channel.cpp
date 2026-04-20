#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <complex>
#include <string>
#include <algorithm>

class RadioChannel {
public:
    static std::vector<std::complex<float>> apply_awgn(const std::vector<std::complex<float>>& signal, float snr_db) {
        std::vector<std::complex<float>> noisy_signal = signal;
        float snr_linear = std::pow(10.0f, snr_db / 10.0f);
        float signal_power = 1.0f;
        float noise_power = signal_power / snr_linear;
        float std_dev = std::sqrt(noise_power / 2.0f);

        std::random_device rd;
        std::default_random_engine generator(rd());
        std::normal_distribution<float> distribution(0.0, std_dev);

        for (std::complex<float>& sample : noisy_signal) {
            sample += std::complex<float>(distribution(generator), distribution(generator));
        }
        return noisy_signal;
    }

    static std::vector<std::complex<float>> apply_rayleigh(const std::vector<std::complex<float>>& signal, float snr_db) {
        std::vector<std::complex<float>> faded(signal.size());
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);

        for (size_t n = 0; n < signal.size(); ++n) {
            std::complex<float> h(normal_dist(generator), normal_dist(generator));
            h /= std::sqrt(2.0f);
            faded[n] = h * signal[n];
        }
        return apply_awgn(faded, snr_db);
    }

    static std::vector<std::complex<float>> apply_doppler(
        const std::vector<std::complex<float>>& signal, float fd_hz, float sample_rate_hz) {
        std::vector<std::complex<float>> shifted(signal.size());
        const float two_pi = 2.0f * static_cast<float>(M_PI);
        for (size_t n = 0; n < signal.size(); ++n) {
            float phase = two_pi * fd_hz * static_cast<float>(n) / sample_rate_hz;
            std::complex<float> rot(std::cos(phase), std::sin(phase));
            shifted[n] = signal[n] * rot;
        }
        return shifted;
    }

    static std::vector<std::complex<float>> apply_multipath(
        const std::vector<std::complex<float>>& signal,
        const std::vector<std::complex<float>>& taps = {
            {0.8f, 0.0f}, {0.4f, 0.2f}, {0.2f, -0.1f}
        }) {
        std::vector<std::complex<float>> out(signal.size(), {0.0f, 0.0f});
        for (size_t n = 0; n < signal.size(); ++n) {
            for (size_t d = 0; d < taps.size(); ++d) {
                if (n >= d) {
                    out[n] += taps[d] * signal[n - d];
                }
            }
        }
        return out;
    }
};

int main(int argc, char** argv) {
    std::string channel = "awgn";
    float snr_db = 10.0f;
    int symbols = 1000;
    float fd_hz = 50.0f;
    float sample_rate_hz = 1000.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--channel" && i + 1 < argc) {
            channel = argv[++i];
        } else if (arg == "--snr" && i + 1 < argc) {
            snr_db = std::stof(argv[++i]);
        } else if (arg == "--symbols" && i + 1 < argc) {
            symbols = std::stoi(argv[++i]);
        } else if (arg == "--fd" && i + 1 < argc) {
            fd_hz = std::stof(argv[++i]);
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            sample_rate_hz = std::stof(argv[++i]);
        }
    }

    std::vector<std::complex<float>> tx(symbols, {1.0f, 0.0f});
    std::vector<std::complex<float>> rx;

    if (channel == "awgn") {
        rx = RadioChannel::apply_awgn(tx, snr_db);
    } else if (channel == "rayleigh") {
        rx = RadioChannel::apply_rayleigh(tx, snr_db);
    } else if (channel == "multipath") {
        auto mp = RadioChannel::apply_multipath(tx);
        rx = RadioChannel::apply_awgn(mp, snr_db);
    } else if (channel == "rayleigh_doppler") {
        auto doppler = RadioChannel::apply_doppler(tx, fd_hz, sample_rate_hz);
        auto faded = RadioChannel::apply_rayleigh(doppler, snr_db);
        rx = RadioChannel::apply_multipath(faded);
    } else {
        std::cerr << "Unknown channel type: " << channel << std::endl;
        std::cerr << "Use awgn, rayleigh, multipath, or rayleigh_doppler" << std::endl;
        return 1;
    }

    std::cout << "HERMES C++ DSP Engine: Channel Simulator Active" << std::endl;
    std::cout << "channel=" << channel << " snr_db=" << snr_db
              << " symbols=" << symbols << " fd_hz=" << fd_hz << std::endl;
    std::cout << "First 5 received symbols:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, rx.size()); ++i) {
        std::cout << i << ": " << rx[i].real() << " + j" << rx[i].imag() << std::endl;
    }
    return 0;
}