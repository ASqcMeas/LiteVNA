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
        # Load fitting mode from configurations
        fitting_config = self.config_manager.vna_config.get("fitting", {})
        self.fitting_mode = fitting_config.get("mode", "refined").strip().lower()

    def _fit_single_port(self, port, fixed_delay=None, fixed_Qc=None, sample_power_dBm=None):
        """
        Fits a single port (notch or reflection) using either free parameters or fixed parameters.
        Matches QCAT's refined fitting approach: only delay and Qc are optionally locked;
        background amplitude (A) and phase offset (alpha) are always auto-calibrated per scan.
        """
        is_reflection = isinstance(port, circuit.reflection_port)
        ignoreslope = True
        guessdelay = not is_reflection
        
        # 1. Perform do_calibration to estimate parameters
        delay_auto, amp_norm_auto, alpha_auto, fr_auto, Ql_auto, A2, frcal = port.do_calibration(
            port.f_data, port.z_data_raw,
            ignoreslope=ignoreslope,
            guessdelay=guessdelay,
            fixed_delay=fixed_delay
        )
        
        # Override only delay if fixed_delay is provided; background amp and alpha remain auto-calibrated
        delay = fixed_delay if fixed_delay is not None else delay_auto
        amp_norm = amp_norm_auto
        alpha = alpha_auto
        
        # 2. Perform do_normalization
        port.z_data = port.do_normalization(
            port.f_data, port.z_data_raw, delay, amp_norm, alpha, A2, frcal
        )
        
        # 3. Perform circlefit on normalized data
        port.fitresults = port.circlefit(
            port.f_data,
            port.z_data,
            fr_auto,
            Ql_auto,
            refine_results=False,
            calc_errors=True
        )
        
        # Save baseline/calibration parameters into fitresults dict for reporting
        port.fitresults["delay"] = delay
        port.fitresults["A"] = amp_norm
        port.fitresults["alpha"] = alpha
        port.fitresults["A2"] = A2
        port.fitresults["frcal"] = frcal
        
        # 4. Generate simulated S-parameter curves (raw and normalized)
        if is_reflection:
            port.z_data_sim = A2 * (port.f_data - frcal) + port._S11_directrefl(
                port.f_data,
                fr=port.fitresults["fr"],
                Ql=port.fitresults["Ql"],
                Qc=port.fitresults["Qc"],
                a=amp_norm,
                alpha=alpha,
                delay=delay,
            )
            port.z_data_sim_norm = port._S11_directrefl(
                port.f_data,
                fr=port.fitresults["fr"],
                Ql=port.fitresults["Ql"],
                Qc=port.fitresults["Qc"],
                a=1.0,
                alpha=0.0,
                delay=0.0,
            )
        else:
            port.z_data_sim = A2 * (port.f_data - frcal) + port._S21_notch(
                port.f_data,
                fr=port.fitresults["fr"],
                Ql=port.fitresults["Ql"],
                Qc=port.fitresults["absQc"],
                phi=port.fitresults["phi0"],
                a=amp_norm,
                alpha=alpha,
                delay=delay,
            )
            port.z_data_sim_norm = port._S21_notch(
                port.f_data,
                fr=port.fitresults["fr"],
                Ql=port.fitresults["Ql"],
                Qc=port.fitresults["absQc"],
                phi=port.fitresults["phi0"],
                a=1.0,
                alpha=0.0,
                delay=0.0,
            )
        port._delay = delay
        
        # 5. Apply fixed Qc if requested (saved in Qi_dia_corr_fqc as in QCAT, leaving original Qi_dia_corr untouched)
        if fixed_Qc is not None:
            port.fitresults["Qi_dia_corr_fqc"] = 1.0 / (1.0 / port.fitresults["Ql"] - 1.0 / fixed_Qc)
            port.fitresults["Qc_dia_corr_fixed"] = fixed_Qc
                
        # 6. Calculate average photons inside resonator
        if sample_power_dBm is not None:
            try:
                # get_photons_in_resonator accepts power in dBm and returns photon count
                port.fitresults["photons"] = port.get_photons_in_resonator(sample_power_dBm)
            except Exception as pe:
                print(f"      Warning: Failed to calculate photons: {pe}")
                port.fitresults["photons"] = np.nan
        else:
            port.fitresults["photons"] = np.nan
            
        return port

    def _plot_resonator_fitting_overlay(self, resonator, loaded_groups, fitting_mode, output_path):
        """
        Generates a unified overlay plot (Amplitude, Phase, and IQ) containing all power levels,
        matching QCAT's plot_resonatorFitting layout.
        """
        import matplotlib.gridspec as gridspec
        
        # Sort groups by power from low to high for consistent rainbow mapping
        sorted_groups = sorted(loaded_groups, key=lambda x: x["power"])
        
        fig = plt.figure(facecolor='white', figsize=(20, 9))
        gs = gridspec.GridSpec(2, 2)
        
        ax_amp = plt.subplot(gs[0, 0])
        ax_amp.set_ylabel("Amplitude")
        ax_amp.locator_params(tight=True)
        
        ax_pha = plt.subplot(gs[1, 0])
        ax_pha.set_xlabel("Frequency (GHz)")
        ax_pha.set_ylabel("Phase (rad)")
        ax_pha.locator_params(tight=True)
        
        ax_iq = plt.subplot(gs[0:, 1])
        ax_iq.set_xlabel("In-phase")
        ax_iq.set_ylabel("Quadrature")
        ax_iq.locator_params(tight=True)
        ax_iq.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax_iq.yaxis.set_major_locator(plt.MaxNLocator(5))
        
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.25, hspace=0.25)
        
        # Generate colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_groups)))
        
        for g, c in zip(sorted_groups, colors):
            port = g["port"]
            freq_ghz = port.f_data * 1e-9
            
            # Amplitude subplot
            ax_amp.plot(freq_ghz, np.abs(port.z_data_raw), 'o', ms=1, color=c)
            ax_amp.plot(freq_ghz, np.abs(port.z_data_sim), '-', linewidth=1, color=c)
            
            # Phase subplot
            ax_pha.plot(freq_ghz, np.unwrap(np.angle(port.z_data_raw)), 'o', ms=1, color=c)
            ax_pha.plot(freq_ghz, np.unwrap(np.angle(port.z_data_sim)), '-', linewidth=1, color=c)
            
            # IQ subplot
            ax_iq.plot(port.z_data_raw.real, port.z_data_raw.imag, 'o', ms=1, color=c)
            ax_iq.plot(port.z_data_sim.real, port.z_data_sim.imag, '-', linewidth=1, color=c, label=f"{g['power']} dBm")
            
        ax_iq.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        fig.suptitle(f"{resonator} - Unified Resonance Fit Overlay ({fitting_mode.upper()} mode)", fontsize=16)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_single_res_powerQ(self, resonator, df, mode, output_path):
        """
        Plots Quality factor (Qi, Qc, Ql) vs Intracavity Photon Number (log-log plot with error bars).
        Matches plot_singleRes_powerQ_free / plot_singleRes_powerQ_refined in QCAT.
        """
        plt.figure(figsize=(8, 6))
        
        photons = df["photons"].values
        ql = df["QL"].values
        ql_err = df["QL_err"].values if "QL_err" in df.columns else np.zeros_like(ql)
        
        if mode == "refined":
            qi = df["Qi_dia_corr_fqc"].values if "Qi_dia_corr_fqc" in df.columns else df["Qi_dia_corr"].values
            qc = df["Qc_dia_corr_fixed"].values if "Qc_dia_corr_fixed" in df.columns else df["Qc_dia_corr"].values
            qi_label, qc_label = "Internal Q (fixed)", "Coupling Q (fixed)"
        else:
            qi = df["Qi_dia_corr"].values
            qc = df["Qc_dia_corr"].values
            qi_label, qc_label = "Internal Q", "Coupling Q"
            
        qi_err = df["Qi_dia_corr_err"].values if "Qi_dia_corr_err" in df.columns else np.zeros_like(qi)
        qc_err = df["Qc_dia_corr_err"].values if "Qc_dia_corr_err" in df.columns else np.zeros_like(qc)
        
        # Filter out invalid indices (negative/zero/NaN) for cleaner plot
        valid = (photons > 0)
        
        plt.errorbar(photons[valid], qi[valid], yerr=qi_err[valid], fmt="o", ms=5, capsize=3, label=qi_label, color="#2ecc71")
        plt.errorbar(photons[valid], qc[valid], yerr=qc_err[valid], fmt="o", ms=5, capsize=3, label=qc_label, color="#e74c3c")
        plt.errorbar(photons[valid], ql[valid], yerr=ql_err[valid], fmt="o", ms=5, capsize=3, label="Loaded Q", color="#3498db")
        
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Intracavity Photon Number $N_{\\mathrm{photons}}$")
        plt.ylabel("Quality Factor")
        plt.title(f"Power-dependent Quality Factors - {resonator} ({mode.upper()})")
        plt.grid(True, which="both", alpha=0.15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _plot_single_res_powerloss(self, resonator, df, mode, output_path):
        """
        Plots Loss (1/Qi, 1/Qc, 1/Ql) vs Intracavity Photon Number (log-log plot with error bars).
        Matches plot_singleRes_powerloss_free / plot_singleRes_powerloss_refined in QCAT.
        """
        plt.figure(figsize=(8, 6))
        
        photons = df["photons"].values
        ql = df["QL"].values
        ql_err = df["QL_err"].values if "QL_err" in df.columns else np.zeros_like(ql)
        
        if mode == "refined":
            qi = df["Qi_dia_corr_fqc"].values if "Qi_dia_corr_fqc" in df.columns else df["Qi_dia_corr"].values
            qc = df["Qc_dia_corr_fixed"].values if "Qc_dia_corr_fixed" in df.columns else df["Qc_dia_corr"].values
            qi_label, qc_label = "Internal Loss (fixed)", "Coupling Loss (fixed)"
        else:
            qi = df["Qi_dia_corr"].values
            qc = df["Qc_dia_corr"].values
            qi_label, qc_label = "Internal Loss", "Coupling Loss"
            
        qi_err = df["Qi_dia_corr_err"].values if "Qi_dia_corr_err" in df.columns else np.zeros_like(qi)
        qc_err = df["Qc_dia_corr_err"].values if "Qc_dia_corr_err" in df.columns else np.zeros_like(qc)
        
        # Calculate loss and propagated error
        # Propagation of error: delta_y = delta_x / x^2
        qi_loss = 1.0 / qi
        qi_loss_err = qi_err / (qi**2)
        
        qc_loss = 1.0 / qc
        qc_loss_err = qc_err / (qc**2)
        
        ql_loss = 1.0 / ql
        ql_loss_err = ql_err / (ql**2)
        
        # Filter out invalid indices (negative/zero/NaN) for cleaner plot
        valid = (photons > 0)
        
        plt.errorbar(photons[valid], qi_loss[valid], yerr=qi_loss_err[valid], fmt="o", ms=5, capsize=3, label=qi_label, color="#2ecc71")
        plt.errorbar(photons[valid], qc_loss[valid], yerr=qc_loss_err[valid], fmt="o", ms=5, capsize=3, label=qc_label, color="#e74c3c")
        plt.errorbar(photons[valid], ql_loss[valid], yerr=ql_loss_err[valid], fmt="o", ms=5, capsize=3, label="Loaded Loss", color="#3498db")
        
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Intracavity Photon Number $N_{\\mathrm{photons}}$")
        plt.ylabel("Loss $\\delta = 1/Q$")
        plt.title(f"Power-dependent Losses - {resonator} ({mode.upper()})")
        plt.grid(True, which="both", alpha=0.15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def run_batch_fitting(self):
        """
        Fits all measured resonators and plots curves, exporting summary CSVs.
        Supports both Free and Refined fitting modes.
        """
        print("\n" + "="*60)
        print("PHASE 5: RUN BATCH RESONANCE FITTING")
        print(f"Fitting Mode: {self.fitting_mode.upper()}")
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
                
            print(f"\nFitting resonator: {resonator} ({len(nc_files)} nc files)")
            
            # Group nc files by parent directory (each directory represents a power level)
            groups = {}
            for file_path in nc_files:
                parent_dir = os.path.dirname(file_path)
                if parent_dir not in groups:
                    groups[parent_dir] = []
                groups[parent_dir].append(file_path)
                
            fit_data_list = []
            heatmap_data = []
            
            # Store loaded data and ports for two-stage fitting
            loaded_groups = []
            
            # Pre-load, average, and run Stage 1 (Free Fit)
            for parent_dir in sorted(groups.keys()):
                group_files = sorted(groups[parent_dir])
                folder_name = os.path.basename(parent_dir)
                
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
                    
                    # Select fitting model based on port type
                    if port_str in ["S11", "S22", "S33", "S44"]:
                        port = circuit.reflection_port(f_data=freq, z_data_raw=complex_s)
                    else:
                        port = circuit.notch_port(f_data=freq, z_data_raw=complex_s)
                        
                    # Run free fit
                    self._fit_single_port(port, sample_power_dBm=p_sample)
                    
                    loaded_groups.append({
                        "parent_dir": parent_dir,
                        "folder_name": folder_name,
                        "group_files": group_files,
                        "freq": freq,
                        "complex_s": complex_s,
                        "power": power,
                        "attenuation": attenuation,
                        "p_sample": p_sample,
                        "port_str": port_str,
                        "port": port
                    })
                except Exception as e:
                    print(f"    Failed loading/free fitting folder {folder_name}: {e}")

            if not loaded_groups:
                continue

            # Stage 1: Select calibration parameters with minimum chi_square
            fixed_delay, fixed_Qc = None, None
            best_group = None
            min_chi = np.inf
            
            for g in loaded_groups:
                p = g["port"]
                chi = p.fitresults.get("chi_square", np.nan)
                if not np.isnan(chi) and chi > 0 and chi < min_chi:
                    min_chi = chi
                    best_group = g
                    
            if best_group is not None:
                p_best = best_group["port"]
                fixed_delay = p_best.fitresults.get("delay", p_best._delay)
                if isinstance(p_best, circuit.reflection_port):
                    fixed_Qc = p_best.fitresults.get("Qc", np.nan)
                else:
                    fixed_Qc = p_best.fitresults.get("Qc_dia_corr", p_best.fitresults.get("absQc", np.nan))
                    
                print(f"  [Refined Params Locked from {best_group['folder_name']} (min ChiSq={min_chi:.5f})]")
                print(f"    delay: {fixed_delay:.3e}, Qc: {fixed_Qc:.3e}")
            else:
                print("  Warning: No valid free fit found. Falling back to free fitting mode for all folders.")
                self.fitting_mode = "free"

            # Stage 2: Execute fitting and save plots
            for g in loaded_groups:
                parent_dir = g["parent_dir"]
                folder_name = g["folder_name"]
                group_files = g["group_files"]
                freq = g["freq"]
                complex_s = g["complex_s"]
                power = g["power"]
                attenuation = g["attenuation"]
                p_sample = g["p_sample"]
                port_str = g["port_str"]
                port = g["port"]
                
                print(f"  Processing folder: {folder_name} (average of {len(group_files)} repeats)...")
                
                try:
                    # Run refined fit if requested and parameter locking was successful
                    if self.fitting_mode == "refined" and best_group is not None:
                        self._fit_single_port(
                            port,
                            fixed_delay=fixed_delay,
                            fixed_Qc=fixed_Qc,
                            sample_power_dBm=p_sample
                        )
                    
                    # Collect magnitude data for heatmap
                    mag_db = 20 * np.log10(np.maximum(np.abs(complex_s), 1e-18))
                    heatmap_data.append((power, freq, mag_db))
                    
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
                    
                    # Display full measured dataset frequency range so baseline is always fully visible
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
                    is_reflection = isinstance(port, circuit.reflection_port)
                    if is_reflection:
                        qi_val = port.fitresults.get("Qi", np.nan)
                        qc_val = port.fitresults.get("Qc", np.nan)
                        qi_dia_corr = qi_val
                        qc_dia_corr = qc_val
                    else:
                        qi_val = port.fitresults.get("Qi_dia_corr", np.nan)
                        qc_val = port.fitresults.get("absQc", np.nan)
                        qi_dia_corr = port.fitresults.get("Qi_dia_corr", np.nan)
                        qc_dia_corr = port.fitresults.get("Qc_dia_corr", np.nan)
                        
                    ql_val = port.fitresults.get("Ql", np.nan)
                    fr_val = port.fitresults.get("fr", np.nan)
                    chi_val = port.fitresults.get("chi_square", np.nan)
                    qi_err = port.fitresults.get("Qi_dia_corr_err", port.fitresults.get("Qi_err", np.nan))
                    ql_err = port.fitresults.get("Ql_err", np.nan)
                    qc_err = port.fitresults.get("absQc_err", port.fitresults.get("Qc_err", np.nan))
                    photons = port.fitresults.get("photons", np.nan)
                    
                    qi_dia_corr_fqc = port.fitresults.get("Qi_dia_corr_fqc", np.nan)
                    qc_dia_corr_fixed = port.fitresults.get("Qc_dia_corr_fixed", np.nan)
                    
                    fit_entry = {
                        "File": f"{folder_name}_averaged",
                        "VNA_Power_dBm": power,
                        "Attenuation_dB": attenuation,
                        "Sample_Power_dBm": p_sample,
                        "f0_GHz": fr_val / 1e9 if not np.isnan(fr_val) else np.nan,
                        "fr": fr_val,
                        "QL": ql_val,
                        "QL_err": ql_err,
                        "Qi": qi_val,
                        "Qc": qc_val,
                        "Qi_dia_corr": qi_dia_corr,
                        "Qc_dia_corr": qc_dia_corr,
                        "Qi_dia_corr_fqc": qi_dia_corr_fqc,
                        "Qc_dia_corr_fixed": qc_dia_corr_fixed,
                        "Qi_dia_corr_err": qi_err,
                        "Qc_dia_corr_err": qc_err,
                        "photons": photons,
                        "fitting_mode": self.fitting_mode,
                        "Fit_ChiSq": chi_val
                    }
                    fit_data_list.append(fit_entry)
                    
                    # Append to global list with resonator identification
                    global_entry = copy.deepcopy(fit_entry)
                    global_entry["Resonator"] = resonator
                    global_fit_data_list.append(global_entry)
                    
                    display_qi = qi_dia_corr_fqc if not np.isnan(qi_dia_corr_fqc) else qi_dia_corr
                    print(f"      Fit succeeded: {folder_name} at {power} dBm (Qi: {display_qi:.1f}, photons: {photons:.2e})")
                except Exception as e:
                    print(f"      Fit failed for folder {folder_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    
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
                    
            # Generate the unified resonance fit overlay plot (matching QCAT's plot_resonatorFitting)
            if len(loaded_groups) > 0:
                try:
                    overlay_path = os.path.join(resonator_dir, f"{resonator}_{self.fitting_mode}_overlay.png")
                    self._plot_resonator_fitting_overlay(resonator, loaded_groups, self.fitting_mode, overlay_path)
                    print(f"    Saved multi-power overlay plot to: {overlay_path}")
                except Exception as oe:
                    print(f"    Failed to generate overlay plot for {resonator}: {oe}")
                    
            if fit_data_list:
                df = pd.DataFrame(fit_data_list)
                df = df.sort_values(by="Sample_Power_dBm")
                output_csv = os.path.join(resonator_dir, f"{resonator}_fit_summary.csv")
                df.to_csv(output_csv, index=False)
                print(f"Saved fit summary sheet to: {output_csv}")
                
                # Plot Free fit Quality Factors and Losses (matching plot_singleRes_powerQ_free / plot_singleRes_powerloss_free)
                try:
                    free_q_path = os.path.join(resonator_dir, f"{resonator}_powerQ_free.png")
                    self._plot_single_res_powerQ(resonator, df, "free", free_q_path)
                    
                    free_loss_path = os.path.join(resonator_dir, f"{resonator}_powerloss_free.png")
                    self._plot_single_res_powerloss(resonator, df, "free", free_loss_path)
                    print(f"    Saved free Q/loss vs photons plots.")
                except Exception as fe:
                    print(f"    Failed to generate free Q/loss plots: {fe}")
                    
                # Plot Refined fit Quality Factors and Losses (matching plot_singleRes_powerQ_refined / plot_singleRes_powerloss_refined)
                if self.fitting_mode == "refined":
                    try:
                        refined_q_path = os.path.join(resonator_dir, f"{resonator}_powerQ_refined.png")
                        self._plot_single_res_powerQ(resonator, df, "refined", refined_q_path)
                        
                        refined_loss_path = os.path.join(resonator_dir, f"{resonator}_powerloss_refined.png")
                        self._plot_single_res_powerloss(resonator, df, "refined", refined_loss_path)
                        print(f"    Saved refined Q/loss vs photons plots.")
                    except Exception as re:
                        print(f"    Failed to generate refined Q/loss plots: {re}")
                
                show_cols = ["Sample_Power_dBm", "f0_GHz", "QL", "Qi_dia_corr", "Qc_dia_corr", "photons"]
                if "Qi_dia_corr_fqc" in df.columns and not df["Qi_dia_corr_fqc"].isna().all():
                    show_cols = ["Sample_Power_dBm", "f0_GHz", "QL", "Qi_dia_corr", "Qc_dia_corr", "Qi_dia_corr_fqc", "photons"]
                print(df[show_cols].to_string(index=False))
                
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
