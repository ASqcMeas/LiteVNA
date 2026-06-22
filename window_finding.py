import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tomlkit
from scipy.signal import find_peaks, peak_widths
from os.path import exists

# 1. Load configuration
config_path = r'LiteMWSInstr/measurement_window_finding.toml' if exists(r'LiteMWSInstr/measurement_window_finding.toml') else 'measurement_window_finding.toml'

if not exists(config_path):
    print(f"Error: Config file '{config_path}' not found.")
    sys.exit(1)

print(f"Loading configuration from: {config_path}")
with open(config_path, 'r', encoding='utf-8') as file:
    content = file.read()
    sweepLF_config = tomlkit.parse(content)

vna_address = sweepLF_config["hardware"]["address"]
vna_model = sweepLF_config["hardware"]["model"]
vna_port = sweepLF_config["hardware"]["port"]
measurements = sweepLF_config["measurement"]
attenuation = sweepLF_config["hardware"]["attenuation"]

# 2. Connect with VNA
from driver import get_VNA
print(f"Connecting to VNA at {vna_address}...")
vna = get_VNA(vna_address, vna_model)
vna.check_error()

def measure_and_find_peak(port, start, stop, points, power, IF_bandwidth):
    print(f"Measuring range: {start/1e9:.6f} GHz to {stop/1e9:.6f} GHz...")
    # Perform a frequency sweep
    freq_array, s_params = vna.lin_freq_sweep(start, stop, points, port, power, IF_bandwidth)
    
    # Extract magnitude from S-parameters
    magnitude = 20 * np.log10(np.abs(s_params))
    
    # Find peaks (valleys/dips in S21 amplitude)
    # We find peaks of -magnitude to locate the dips
    peaks, _ = find_peaks(-magnitude)  

    if len(peaks) > 0:
        # Pick the deepest dip
        peak_index = peaks[np.argmin(magnitude[peaks])]
        peak_frequency = freq_array[peak_index]
        print(f"Found dip at: {peak_frequency/1e9:.6f} GHz ({magnitude[peak_index]:.2f} dB)")

        # Find the full width at half maximum (FWHM) of the peak
        results_half = peak_widths(-magnitude, [peak_index], rel_height=0.5)
        fwhm = results_half[0][0] * (freq_array[1] - freq_array[0])
        print(f"FWHM: {fwhm/1e6:.3f} MHz")

        # Determine new start and stop frequencies based on the peak and FWHM
        new_start = peak_frequency - 15 * fwhm
        new_stop = peak_frequency + 15 * fwhm

        return new_start, new_stop
    else:
        raise ValueError("No downward dip found in the specified range.")

resonatorfre = []

# 3. Execution Loop
try:
    for m_task in measurements:
        print("\n" + "="*50)
        deltafre = m_task['frequency']['deltafre']
        resonator_fre = m_task["frequency"]["resonator_fre"]
        sweep_point = m_task["frequency"]["points"]
        vna_power = m_task["power"]
        IF_bandwidth = m_task["IF_bandwidth"]

        start_freq = resonator_fre - deltafre
        stop_freq = resonator_fre + deltafre
        
        try:
            # Step A: Perform the initial wide measurement and find the peak
            print(f"Step A: Coarse sweep around {resonator_fre/1e9:.5f} GHz...")
            new_start, new_stop = measure_and_find_peak(vna_port, start_freq, stop_freq, sweep_point, vna_power, IF_bandwidth)
            
            # Step B: Adjust the frequency range and measure again for fine tuning
            print("Step B: Fine sweep within optimized window...")
            new_start, new_stop = measure_and_find_peak(vna_port, new_start, new_stop, sweep_point, vna_power, IF_bandwidth)
            
            resonatorfre.append((new_start, new_stop, resonator_fre))
        except ValueError as e:
            print(f"Failed to find dip near {resonator_fre/1e9:.5f} GHz: {e}")
            
    # 4. Final verification & plotting
    if len(resonatorfre) > 0:
        print("\n" + "="*50)
        print("Final Verification & Plotting...")
        print("="*50)
        for idx, (new_start, new_stop, original_fre) in enumerate(resonatorfre):
            print(f"Refined search window for original frequency {original_fre/1e9:.5f} GHz:")
            print(f"  Start: {new_start/1e9:.6f} GHz")
            print(f"  Stop:  {new_stop/1e9:.6f} GHz")
            
            # Final verification sweep
            freq_array, s_params = vna.lin_freq_sweep(new_start, new_stop, 501, vna_port, vna_power, 1000)
            
            # Plot the results
            freq_ghz = freq_array / 1e9
            magnitude_db = 20 * np.log10(np.abs(s_params))
            phase_rad = np.angle(s_params)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
            
            # Plot Magnitude
            ax1.plot(freq_ghz, magnitude_db, color='#1f77b4', linewidth=2, label='Magnitude')
            ax1.set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend(loc='upper right')
            ax1.set_title(f"Refined Window Verification - Resonator C_{int(original_fre/1e6)}", fontsize=12, fontweight='bold')
            
            # Plot Phase
            ax2.plot(freq_ghz, phase_rad, color='#ff7f0e', linewidth=2, label='Phase')
            ax2.set_xlabel('Frequency (GHz)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Phase (rad)', fontsize=11, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            print("Opening plot window... (Close the window to proceed)")
            plt.show()

finally:
    # Clean up connection
    print("\nDisconnecting from VNA...")
    try:
        vna.disconnect()
    except Exception as e:
        print(f"Error disconnecting: {e}")
