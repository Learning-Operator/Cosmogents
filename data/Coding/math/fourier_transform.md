


# Fourier Transform

The Fourier Transform is a mathematical transform that decomposes a function (often a signal or waveform) into its constituent frequencies. It provides a way to represent a time-domain signal in the frequency domain. This is especially useful in fields such as signal processing, physics, and engineering.

## 1. Fourier Transform (Continuous)

The Fourier Transform \( \mathcal{F}\{f(t)\} \) of a continuous function \( f(t) \) is defined as:

\[
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
\]

Where:
- \( f(t) \) is the function in the time domain.
- \( F(\omega) \) is the Fourier Transform of \( f(t) \), representing the frequency domain.
- \( \omega \) is the angular frequency, defined as \( \omega = 2\pi f \), where \( f \) is the frequency in Hz.
- \( i \) is the imaginary unit.

### Inverse Fourier Transform

To recover the original function \( f(t) \) from its Fourier Transform \( F(\omega) \), we use the inverse Fourier Transform:

\[
f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} d\omega
\]

Where:
- \( f(t) \) is the original time-domain function.
- \( F(\omega) \) is the Fourier Transform of the function.
- \( \omega \) is the angular frequency.

