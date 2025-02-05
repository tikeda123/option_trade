import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def simulate_gbm(S0, mu, sigma, T, N):
    """
    Stock price simulation using Geometric Brownian Motion (GBM)
    """
    dt = T / N
    time = np.linspace(0, T, N)
    # Generate Brownian motion
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.cumsum(dW)
    # Calculate price using geometric Brownian motion
    S = S0 * np.exp((mu - 0.5 * sigma**2) * time + sigma * W)
    return time, S

def detect_elliott_5wave(S, threshold=0.02):
    """
    A heuristic to detect "5-wave pattern" (6 consecutive extrema showing Up-Down-Up-Down-Up,
    or Down-Up-Down-Up-Down).

    threshold: minimum relative change rate between adjacent extrema
    """
    # Detect local maxima and minima
    # Note: find_peaks without parameters will also detect noise-like peaks
    peaks, _ = find_peaks(S)
    troughs, _ = find_peaks(-S)

    # Combine and sort all extrema indices chronologically
    extrema = np.sort(np.concatenate([peaks, troughs]))
    if len(extrema) < 6:
        return False, None  # Cannot form 5 waves with less than 6 points

    # Check 6 extrema (5 intervals) consecutively
    # Example: For values [e0, e1, e2, e3, e4, e5],
    # Wave 1: e0 -> e1, Wave 2: e1 -> e2, Wave 3: e2 -> e3, Wave 4: e3 -> e4, Wave 5: e4 -> e5
    # Check if they alternate up/down and exceed threshold amplitude
    for i in range(len(extrema) - 5):
        seg_idx = extrema[i:i+6]
        seg_val = S[seg_idx]

        # First, determine if seg_val[0] -> seg_val[1] is rising or falling
        # Then check alternating pattern (Up->Down->Up->Down->Up or Down->Up->Down->Up->Down)
        upward_start = (seg_val[1] > seg_val[0])  # True for Up start, False for Down start

        # Check Up/Down pattern for 5 intervals
        pattern_ok = True
        for w in range(5):
            if upward_start:
                # For even w: rising, for odd w: falling
                if w % 2 == 0:
                    # Check rise
                    if not (seg_val[w+1] > seg_val[w]):
                        pattern_ok = False
                        break
                else:
                    # Check fall
                    if not (seg_val[w+1] < seg_val[w]):
                        pattern_ok = False
                        break
            else:
                # For even w: falling, for odd w: rising
                if w % 2 == 0:
                    # Check fall
                    if not (seg_val[w+1] < seg_val[w]):
                        pattern_ok = False
                        break
                else:
                    # Check rise
                    if not (seg_val[w+1] > seg_val[w]):
                        pattern_ok = False
                        break

        if not pattern_ok:
            continue  # Move to next interval

        # Amplitude check: verify if each wave's change rate exceeds threshold
        # Example: Wave 1 change rate = |(seg_val[1] - seg_val[0]) / seg_val[0]|
        wave_moves = []
        for w in range(5):
            # Add +1e-8 to handle potential zero denominator
            base = seg_val[w] if seg_val[w] != 0 else 1e-8
            wave_moves.append( abs(seg_val[w+1] - seg_val[w]) / abs(base) )

        # Check if all wave movements exceed threshold
        if all(move > threshold for move in wave_moves):
            # Consider pattern found
            return True, seg_idx

    # Pattern not found if we reach here
    return False, None

# --- Simulation and Detection Parameters ---
S0 = 100.0       # Initial stock price
mu = 0.05        # Mean growth rate
sigma = 0.2      # Volatility
T = 1.0          # Simulation period (years)
N = 252          # Number of steps (assuming 252 trading days per year)

# Maximum number of simulation attempts
max_attempts = 5000

np.random.seed(42)  # Set random seed for reproducibility

found = False
attempt = 0

while not found and attempt < max_attempts:
    attempt += 1
    t, S = simulate_gbm(S0, mu, sigma, T, N)

    found, pattern_indices = detect_elliott_5wave(S, threshold=0.02)
    if found:
        print(f"[Attempt {attempt}] Simple 5-wave pattern detected.")
        # Visualize detection results
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, label="GBM Simulation", color="blue")
        plt.scatter(t[pattern_indices], S[pattern_indices], color="red", zorder=5, label="Detected Extrema")
        plt.title("Simple 5-Wave Pattern Detection Example")
        plt.xlabel("Time (years)")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        plt.show()
        break

if not found:
    print(f"No pattern detected after {max_attempts} attempts.")
