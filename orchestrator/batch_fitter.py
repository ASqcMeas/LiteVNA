import os
import glob
import copy
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from resonator_tools import circuit

class BatchFitter:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def run_batch_fitting(self):
        """
        Fits all measured resonators and plots curves, exporting summary CSVs.
        """
        print("\n" + "="*60)
        print("PHASE 5: RUN BATCH RESONANCE FITTING")
        print("="*60)
        
        res_pd_config = self.config_manager.res_pd_config
        base_data_dir = res_pd_config.get("output", {}).get("data_path", "data/raw")
        if not os.path.exists(base_data_dir):
            print(f"Error: Base output directory {base_data_dir} does not exist. No data to fit.")
            return
            
        resonators = [
            d for d in os.listdir(base_data_dir) 
            if os.path.isdir(os.path.join(base_data_dir, d)) and d not in ["nc", "plots"]
        ]
        print(f"Found resonators to fit: {resonators}")
        
        global_fit_data_list = []
        
        for resonator in sorted(resonators):
            resonator_dir = os.path.join(base_data_dir, resonator)
            nc_files = sorted(glob.glob(os.path.join(resonator_dir, "**", "*.nc"), recursive=True))
            if not nc_files:
                continue
                
            print(f"\nFitting resonator: {resonator} ({len(nc_files)} files)")
            
            # Group nc files by parent directory (each directory represents a power level)
            groups = {}
            for file_path in nc_files:
                parent_dir = os.path.dirname(file_path)
                if parent_dir not in groups:
                    groups[parent_dir] = []
                groups[parent_dir].append(file_path)
                
            fit_data_list = []
            heatmap_data = []
            
            for parent_dir in sorted(groups.keys()):
                group_files = sorted(groups[parent_dir])
                folder_name = os.path.basename(parent_dir)
                print(f"  Power folder: {folder_name} ({len(group_files)} repeats)...")
                
                try:
                    freq = None
                    list_complex_s = []
                    power = None
                    attenuation = None
                    port_str = None
                    
                    for file_path in group_files:
                        with xr.open_dataset(file_path) as ds:
                            freq = ds.frequency.values
                            s_param_keys = ["s21", "s11", "s22", "s12", "s33", "s44", "s13", "s31", "s14", "s41", "s23", "s32", "s24", "s42", "s34", "s43"]
                            var_key = "s21"
                            for k in ds.data_vars:
                                if k.lower() in s_param_keys:
                                    var_key = k
                                    break
                            if var_key not in ds.data_vars and len(ds.data_vars) > 0:
                                var_key = list(ds.data_vars.keys())[0]
                                
                            s_param_data = ds[var_key].values
                            list_complex_s.append(s_param_data[0] + 1j * s_param_data[1])
                            
                            if power is None:
                                power = ds.attrs.get("power", np.nan)
                                attenuation = ds.attrs.get("attenuation", 0)
                                port_str = ds.attrs.get("port", var_key).upper()
                    
                    # Compute complex average across all repeats
                    complex_s = np.mean(list_complex_s, axis=0)
                    p_sample = power - attenuation
                    
                    # Collect magnitude data for heatmap
                    mag_db = 20 * np.log10(np.maximum(np.abs(complex_s), 1e-18))
                    heatmap_data.append((power, freq, mag_db))
                    
                    # Select fitting model based on port type
                    if port_str in ["S11", "S22", "S33", "S44"]:
                        print(f"    [Fit] Using Reflection model (reflection_port) for {port_str}...")
                        port = circuit.reflection_port(f_data=freq, z_data_raw=complex_s)
                    else:
                        print(f"    [Fit] Using Transmission model (notch_port) for {port_str}...")
                        port = circuit.notch_port(f_data=freq, z_data_raw=complex_s)
                        
                    port.autofit()
                    
                    # Dynamically center and scale the plotting X-axis around the fitted resonance frequency fr
                    fr_fit = port.fitresults.get("fr", np.nan)
                    ql_fit = port.fitresults.get("Ql", np.nan)
                    
                    # 1. IQ Plot (Re vs Im)
                    plt.figure(figsize=(6, 5))
                    plt.plot(port.z_data_raw.real, port.z_data_raw.imag, '.', label="rawdata(avg)", color="#3498db", alpha=0.6)
                    plt.plot(port.z_data_sim.real, port.z_data_sim.imag, '-', label="fit", color="#e74c3c", linewidth=2)
                    plt.xlabel("Re(S)")
                    plt.ylabel("Im(S)")
                    plt.title(f"{resonator} at {power} dBm - Resonance Circle (Averaged)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(parent_dir, "fit_IQ.png"), dpi=120)
                    plt.close()
                    
                    # Determine dynamic frequency plotting range
                    fr_ghz = fr_fit / 1e9 if not np.isnan(fr_fit) else (freq[0] + freq[-1]) / 2 / 1e9
                    if not np.isnan(fr_fit) and not np.isnan(ql_fit) and ql_fit > 0:
                        fwhm_ghz = fr_ghz / ql_fit
                        half_width_ghz = 15.0 * fwhm_ghz
                        # Guardrail: do not exceed the actual swept frequency range
                        swept_span_ghz = (freq[-1] - freq[0]) / 1e9
                        if half_width_ghz > swept_span_ghz / 2:
                            half_width_ghz = swept_span_ghz / 2
                        xlim_range = (fr_ghz - half_width_ghz, fr_ghz + half_width_ghz)
                    else:
                        xlim_range = (freq[0]/1e9, freq[-1]/1e9)
                        
                    # 2. Amplitude Plot (|S| vs Frequency)
                    plt.figure(figsize=(6, 4))
                    plt.plot(port.f_data * 1e-9, np.absolute(port.z_data_raw), '.', label="rawdata(avg)", color="#3498db", alpha=0.6)
                    plt.plot(port.f_data * 1e-9, np.absolute(port.z_data_sim), '-', label="fit", color="#e74c3c", linewidth=2)
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("|S| Magnitude")
                    plt.xlim(xlim_range)
                    plt.title(f"{resonator} at {power} dBm - Amplitude (Averaged)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(parent_dir, "fit_amplitude.png"), dpi=120)
                    plt.close()
                    
                    # 3. Phase Plot (arg(S) vs Frequency)
                    plt.figure(figsize=(6, 4))
                    plt.plot(port.f_data * 1e-9, np.angle(port.z_data_raw), '.', label="rawdata(avg)", color="#3498db", alpha=0.6)
                    plt.plot(port.f_data * 1e-9, np.angle(port.z_data_sim), '-', label="fit", color="#e74c3c", linewidth=2)
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Phase (rad)")
                    plt.xlim(xlim_range)
                    plt.title(f"{resonator} at {power} dBm - Phase (Averaged)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(parent_dir, "fit_phase.png"), dpi=120)
                    plt.close()
                    
                    # Collect parameters dynamically based on model type
                    if isinstance(port, circuit.reflection_port):
                        qi_val = port.fitresults.get("Qi", np.nan)
                        qc_val = port.fitresults.get("Qc", np.nan)
                    else:
                        qi_val = port.fitresults.get("Qi_dia_corr", np.nan)
                        qc_val = port.fitresults.get("absQc", np.nan)
                        
                    ql_val = port.fitresults.get("Ql", np.nan)
                    fr_val = port.fitresults.get("fr", np.nan)
                    chi_val = port.fitresults.get("chi_square", np.nan)
                    
                    fit_entry = {
                        "File": f"{folder_name}_averaged",
                        "VNA_Power_dBm": power,
                        "Attenuation_dB": attenuation,
                        "Sample_Power_dBm": p_sample,
                        "f0_GHz": fr_val / 1e9 if not np.isnan(fr_val) else np.nan,
                        "QL": ql_val,
                        "Qi": qi_val,
                        "Qc": qc_val,
                        "Fit_ChiSq": chi_val
                    }
                    fit_data_list.append(fit_entry)
                    
                    # Append to global list with resonator identification
                    global_entry = copy.deepcopy(fit_entry)
                    global_entry["Resonator"] = resonator
                    global_fit_data_list.append(global_entry)
                    
                    print(f"    Fit succeeded: {folder_name} (average of {len(group_files)} repeats) at {power} dBm (Qi: {qi_val:.1f})")
                except Exception as e:
                    print(f"    Fit failed for folder {folder_name}: {e}")
                    
            # Generate 2D power-dependent heatmap if we have multiple powers
            if len(heatmap_data) > 1:
                try:
                    heatmap_data = sorted(heatmap_data, key=lambda x: x[0])
                    powers = [x[0] for x in heatmap_data]
                    freq_ghz = heatmap_data[0][1] / 1e9
                    mag_matrix = np.array([x[2] for x in heatmap_data])
                    
                    plt.figure(figsize=(8, 6))
                    mesh = plt.pcolormesh(freq_ghz, powers, mag_matrix, cmap='viridis', shading='auto')
                    cbar = plt.colorbar(mesh)
                    cbar.set_label('|S21| Magnitude (dB)')
                    plt.xlabel('Frequency (GHz)')
                    plt.ylabel('VNA Power (dBm)')
                    plt.ylim(min(powers), max(powers)) # Y-axis from low to high (e.g. -60 to 0)
                    plt.title(f'Power-Dependent Resonance Map - {resonator}')
                    plt.grid(True, alpha=0.15, linestyle=':')
                    plt.tight_layout()
                    
                    heatmap_path = os.path.join(resonator_dir, f"{resonator}_power_heatmap.png")
                    plt.savefig(heatmap_path, dpi=150)
                    plt.close()
                    print(f"    Saved power sweep heatmap to: {heatmap_path}")
                except Exception as he:
                    print(f"    Failed to generate heatmap for {resonator}: {he}")
                    
            if fit_data_list:
                df = pd.DataFrame(fit_data_list)
                df = df.sort_values(by="Sample_Power_dBm")
                output_csv = os.path.join(resonator_dir, f"{resonator}_fit_summary.csv")
                df.to_csv(output_csv, index=False)
                print(f"Saved fit summary sheet to: {output_csv}")
                print(df[["Sample_Power_dBm", "f0_GHz", "QL", "Qi", "Qc"]].to_string(index=False))
                
        # Save global comprehensive fit summary CSV
        if global_fit_data_list:
            global_df = pd.DataFrame(global_fit_data_list)
            # Reorder columns to put Resonator first
            cols = ["Resonator"] + [col for col in global_df.columns if col != "Resonator"]
            global_df = global_df[cols]
            global_df = global_df.sort_values(by=["Resonator", "Sample_Power_dBm"])
            sample_name = res_pd_config.get("sample", {}).get("name", "resonator")
            global_csv_path = os.path.join(base_data_dir, f"{sample_name}_global_fit_summary.csv")
            global_df.to_csv(global_csv_path, index=False)
            print(f"\nSaved global comprehensive fit summary to: {global_csv_path}")
