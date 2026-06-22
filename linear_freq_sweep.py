import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tomlkit
from datetime import datetime
from os import makedirs
from os.path import exists

# 1. Load configuration
# config_path = 'power_dep_resonator.toml'
# To use the wideband sweep config instead, uncomment the line below:'
config_path = r'LiteMWSInstr/measurement_LF.toml' if exists(r'LiteMWSInstr/measurement_LF.toml') else 'measurement_LF.toml'

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

# 3. Measurement Loop
freq_array = None
s_params = None
label = "Measurement"

try:
    for m_idx, m_task in enumerate(measurements):
        print("\n" + "="*50)
        print(f"Task {m_idx + 1}/{len(measurements)}: {m_task.get('label', 'unlabeled')}")
        print(m_task)
        print("="*50)
        
        output_folder = m_task["output"]
        label = m_task.get("label", "C_Unknown")

        freq_start = m_task["frequency"]["start"]
        freq_stop = m_task["frequency"]["stop"]
        sweep_point = m_task["frequency"]["points"]
        vna_power = m_task["power"]
        IF_bandwidth = m_task["IF_bandwidth"]
        repeat = m_task.get("repeat", 1)
        
        for i in range(repeat):
            print(f"Running sweep {i + 1}/{repeat}...")
            start_time = datetime.now()
            
            # Execute VNA linear frequency sweep
            freq_array, s_params = vna.lin_freq_sweep(
                freq_start, freq_stop, sweep_point, vna_port, 
                power=vna_power, IF_bandwith=IF_bandwidth
            )
            
            end_time = datetime.now()
            
            # Create xarray dataset
            output_data = {
                "s21": (["s_params", "frequency"], np.array([s_params.real, s_params.imag]))
            }
            dataset = xr.Dataset(
                output_data,
                coords={"s_params": np.array(["real", "imag"]), "frequency": freq_array}
            )
            
            dataset.attrs["IF_bandwidth"] = IF_bandwidth
            dataset.attrs["power"] = vna_power
            dataset.attrs["attenuation"] = attenuation
            dataset.attrs["start_time"] = str(start_time.strftime("%Y%m%d_%H%M%S"))
            dataset.attrs["end_time"] = str(end_time.strftime("%Y%m%d_%H%M%S"))

            if not exists(output_folder):
                makedirs(output_folder)
                print(f"Created directory: {output_folder}")

            file_path = f"{output_folder}/{label}_{start_time.strftime('%Y%m%d_%H%M%S')}.nc"
            dataset.to_netcdf(file_path)
            print(f"Saved dataset to: {file_path}")

finally:
    # Always disconnect from VNA
    print("Disconnecting from VNA...")
    try:
        vna.disconnect()
    except Exception as e:
        print(f"Error disconnecting: {e}")

# 4. Interactive Plotting (if data was successfully measured)
if freq_array is not None and s_params is not None:
    print("\nGenerating interactive plots...")
    
    freq_ghz = freq_array / 1e9
    magnitude_db = 20 * np.log10(np.abs(s_params))
    phase_rad = np.angle(s_params)
    
    # Use shared x-axis (frequency) so zoom/pan updates both plots simultaneously
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    
    # 1. Magnitude plot (dB)
    ax1.plot(freq_ghz, magnitude_db, color='#1f77b4', linewidth=2, label='S21 Magnitude')
    ax1.set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_title(f'S21 Resonance Transmission - Resonator {label}', fontsize=12, fontweight='bold', pad=15)
    
    # Create dynamic local minimum tracker (marker, vertical lines, and annotation)
    dip_dot, = ax1.plot([], [], 'ro', markersize=6, zorder=5, label='Local Dip')
    dip_text = ax1.annotate("", 
                            xy=(0, 0), 
                            xytext=(15, 15), 
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.5", fc="#FFFFFF", alpha=0.9, ec="#1f77b4", lw=1.5),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#1f77b4", lw=1.5))
    
    # Dual vertical cursor lines across both subplots
    v_line1 = ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.7, visible=False)
    v_line2 = ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.7, visible=False)
    
    # Hide dynamic elements initially
    dip_dot.set_visible(False)
    dip_text.set_visible(False)
    
    def on_move(event):
        if event.inaxes in [ax1, ax2] and event.xdata is not None:
            x_mouse = event.xdata
            
            # Find the index of the closest frequency point to the mouse
            idx = np.argmin(np.abs(freq_ghz - x_mouse))
            
            # Define a local window around the cursor (e.g., ±25 points)
            window_radius = 25
            start_idx = max(0, idx - window_radius)
            end_idx = min(len(freq_ghz), idx + window_radius + 1)
            
            # Find the local minimum (dip) in this neighborhood
            local_min_idx = start_idx + np.argmin(magnitude_db[start_idx:end_idx])
            local_freq = freq_ghz[local_min_idx]
            local_val = magnitude_db[local_min_idx]
            
            # Update marker position
            dip_dot.set_data([local_freq], [local_val])
            dip_dot.set_visible(True)
            
            # Update annotation content and target position
            dip_text.set_text(f"Local Dip:\nFreq: {local_freq:.5f} GHz\nAmp: {local_val:.2f} dB")
            dip_text.xy = (local_freq, local_val)
            dip_text.set_visible(True)
            
            # Update vertical cursor lines on both axes
            v_line1.set_xdata([x_mouse])
            v_line2.set_xdata([x_mouse])
            v_line1.set_visible(True)
            v_line2.set_visible(True)
            
            fig.canvas.draw_idle()
        else:
            # Hide elements when mouse is outside the plot area
            dip_dot.set_visible(False)
            dip_text.set_visible(False)
            v_line1.set_visible(False)
            v_line2.set_visible(False)
            fig.canvas.draw_idle()

    # Connect the motion event to Matplotlib canvas
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    ax1.legend(loc='upper right')
    
    # 2. Phase plot (radians)
    ax2.plot(freq_ghz, phase_rad, color='#ff7f0e', linewidth=2, label='S21 Phase')
    ax2.set_xlabel('Frequency (GHz)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Phase (rad)', fontsize=11, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    print("Opening interactive plotting window... (Close the window to exit the script)")
    plt.show()
else:
    print("No data measured to plot.")
