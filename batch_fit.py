import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
from resonator_tools import circuit

# Base data directory relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "data", "raw")
if not os.path.exists(base_data_dir):
    # Fallback to current directory data/raw if script is run from workspace root
    base_data_dir = "data/raw"

print(f"Searching for resonator data folders in: {os.path.abspath(base_data_dir)}")
resonators = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
print(f"Found resonators: {resonators}")

for resonator in sorted(resonators):
    resonator_dir = os.path.join(base_data_dir, resonator)
    print("\n" + "="*60)
    print(f"Analyzing Resonator: {resonator}")
    print("="*60)
    
    nc_files = sorted(glob.glob(os.path.join(resonator_dir, "**", "*.nc"), recursive=True))
    if not nc_files:
        print(f"No .nc data files found in {resonator_dir}.")
        continue
        
    print(f"Found {len(nc_files)} data files.")
    fit_data_list = []
    
    for file_path in nc_files:
        try:
            # 1. Load data
            ds = xr.open_dataset(file_path)
            freq = ds.frequency.values
            # Reconstruct complex S21: real + i*imag
            s21 = ds.s21.values[0] + 1j * ds.s21.values[1]
            
            # Get metadata attributes
            power = ds.attrs.get("power", np.nan)
            attenuation = ds.attrs.get("attenuation", 0)
            p_sample = power - attenuation # Actual power reaching the sample (dBm)
            
            # 2. Fit notch port
            port = circuit.notch_port(f_data=freq, z_data_raw=s21)
            port.autofit() # Circle fit
            
            # 3. Save fit plot next to the NC file
            plot_name = file_path.replace(".nc", "_fit.png")
            port.plotall()
            plt.savefig(plot_name, dpi=120, bbox_inches='tight')
            plt.close()
            
            # 4. Append metrics
            qi_val = port.fitresults.get("Qi_dia_corr", np.nan)
            ql_val = port.fitresults.get("Ql", np.nan)
            qc_val = port.fitresults.get("absQc", np.nan)
            fr_val = port.fitresults.get("fr", np.nan)
            chi_val = port.fitresults.get("chi_square", np.nan)
            
            fit_data_list.append({
                "File": os.path.basename(file_path),
                "VNA_Power_dBm": power,
                "Attenuation_dB": attenuation,
                "Sample_Power_dBm": p_sample,
                "f0_GHz": fr_val / 1e9 if not np.isnan(fr_val) else np.nan,
                "QL": ql_val,
                "Qi": qi_val,
                "Qc": qc_val,
                "Fit_ChiSq": chi_val
            })
            print(f"  Successfully fit: {os.path.basename(file_path)} at {power} dBm (Qi: {qi_val:.1f})")
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            
    # 5. Export summary sheet for this resonator
    if fit_data_list:
        df = pd.DataFrame(fit_data_list)
        df = df.sort_values(by="Sample_Power_dBm")
        output_csv = os.path.join(resonator_dir, f"{resonator}_fit_summary.csv")
        df.to_csv(output_csv, index=False)
        print(f"\nCompleted analysis for {resonator}. Summary exported to:\n  {output_csv}")
        
        # Display short summary table
        print(df[["Sample_Power_dBm", "f0_GHz", "QL", "Qi", "Qc"]].to_string(index=False))
