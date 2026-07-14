import os
import glob
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TLSAnalyzer:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
        # Load TLS configurations
        tls_config = self.config_manager.vna_config.get("tls_analysis", {})
        self.n_sat_lowerbound = float(tls_config.get("n_sat_lowerbound", 1e-2))
        self.n_sat_upperbound = float(tls_config.get("n_sat_upperbound", 1e4))
        self.alpha_lowerbound = float(tls_config.get("alpha_lowerbound", 0.2))
        self.max_error_threshold = float(tls_config.get("max_error_threshold", 1e8))

    def _model_func(self, N_photons_log, delta_TLS, N_sat, alpha, delta_0):
        N_photons = 10**N_photons_log
        return delta_TLS / np.sqrt(1 + (N_photons / N_sat)**alpha) + delta_0

    def _model_func_fixed_nsat_low(self, N_photons_log, delta_TLS, alpha, delta_0):
        N_photons = 10**N_photons_log
        return delta_TLS / np.sqrt(1 + (N_photons / self.n_sat_lowerbound)**alpha) + delta_0

    def _model_func_fixed_nsat_high(self, N_photons_log, delta_TLS, alpha, delta_0):
        N_photons = 10**N_photons_log
        return delta_TLS / np.sqrt(1 + (N_photons / self.n_sat_upperbound)**alpha) + delta_0

    def _model_func_fixed_alpha(self, N_photons_log, delta_TLS, N_sat, delta_0):
        N_photons = 10**N_photons_log
        return delta_TLS / np.sqrt(1 + (N_photons / N_sat)**self.alpha_lowerbound) + delta_0

    def _filter_data(self, photons, delta_tot, errors):
        # Discard points with negative or zero error, error too large, or negative Qi loss
        valid_indices = np.where((errors > 0) & (errors < self.max_error_threshold) & (delta_tot > 0) & (photons > 0))
        return photons[valid_indices], delta_tot[valid_indices], errors[valid_indices]

    def _find_closest_photon_number(self, photons, delta_tot):
        if len(photons) == 0:
            return 1e0
        delta_max = np.max(delta_tot)
        delta_min = np.min(delta_tot)
        delta_N_c = (delta_max - delta_min) / np.sqrt(2) + delta_min
        closest_index = np.argmin(np.abs(delta_tot - delta_N_c))
        return photons[closest_index]

    def _calculate_rss(self, fitted_values, actual_values, errors):
        residuals = (actual_values - fitted_values) / errors
        return np.sum(residuals**2)

    def run_tls_analysis(self):
        """
        Loads Phase 5 outputs, performs TLS curve fitting, outputs summaries, and plots results.
        """
        print("\n" + "="*60)
        print("PHASE 6: RUN BATCH TLS LOSS FITTING")
        print("="*60)
        
        res_pd_config = self.config_manager.res_pd_config
        base_data_dir = res_pd_config.get("output", {}).get("data_path", "data/raw")
        if not os.path.exists(base_data_dir):
            print(f"Error: Base output directory {base_data_dir} does not exist. No data to analyze.")
            return
            
        resonators = [
            d for d in os.listdir(base_data_dir) 
            if os.path.isdir(os.path.join(base_data_dir, d)) and d not in ["nc", "plots"]
        ]
        
        global_tls_results = []
        
        for resonator in sorted(resonators):
            resonator_dir = os.path.join(base_data_dir, resonator)
            summary_csv = os.path.join(resonator_dir, f"{resonator}_fit_summary.csv")
            if not os.path.exists(summary_csv):
                continue
                
            print(f"\nAnalyzing TLS for resonator: {resonator}")
            try:
                df = pd.read_csv(summary_csv)
                
                # Check required columns. Prioritize Qi_dia_corr_fqc from fixed Qc refined fitting
                qi_col = "Qi_dia_corr_fqc" if "Qi_dia_corr_fqc" in df.columns and not df["Qi_dia_corr_fqc"].isna().all() else ("Qi_dia_corr" if "Qi_dia_corr" in df.columns else "Qi")
                qc_col = "Qc_dia_corr" if "Qc_dia_corr" in df.columns else "Qc"
                qi_err_col = "Qi_dia_corr_err" if "Qi_dia_corr_err" in df.columns else ("Qi_err" if "Qi_err" in df.columns else "")
                
                if qi_col not in df.columns or "photons" not in df.columns:
                    print(f"    Missing required columns in {summary_csv}. Skipping.")
                    continue
                    
                # Extract and sort data by photons
                df_sorted = df.sort_values(by="photons")
                photons = df_sorted["photons"].values
                qi_vals = df_sorted[qi_col].values
                
                # Compute loss delta_tot = 1/Qi
                delta_tot = 1.0 / qi_vals
                
                # Retrieve errors (use uniform weights if error column is missing or empty)
                if qi_err_col and qi_err_col in df_sorted.columns:
                    # delta error approx delta_tot_err = qi_err / qi^2
                    qi_err_vals = df_sorted[qi_err_col].values
                    errors = qi_err_vals / (qi_vals**2)
                else:
                    errors = np.ones_like(delta_tot) * 1e-6
                    
                # Filter bad data points
                p_filtered, d_filtered, e_filtered = self._filter_data(photons, delta_tot, errors)
                if len(p_filtered) < 4:
                    print(f"    Warning: Too few valid data points ({len(p_filtered)}) for TLS fitting. Skipping.")
                    continue
                    
                log_p_filtered = np.log10(p_filtered)
                
                # Initial guesses based on QCAT logic
                # Uses the 1/15th percentile elements to establish baseline/TLS levels
                n_quarter = max(1, len(d_filtered) // 15)
                delta_0_init = d_filtered[-n_quarter]
                delta_TLS_init = d_filtered[n_quarter] - delta_0_init
                if delta_TLS_init <= 0:
                    delta_TLS_init = np.max(d_filtered) - np.min(d_filtered)
                alpha_init = 1.0
                N_sat_init = self._find_closest_photon_number(p_filtered, d_filtered)
                
                # Try multiple N_sat initial guesses to prevent local minima trapping
                N_sat_initials = [0.1 * N_sat_init, N_sat_init, 10 * N_sat_init]
                
                # Extract first valid positive fr in df (mimics QCAT's iloc[0] but filters out failed fits)
                fr_series = df_sorted["fr"] if "fr" in df_sorted.columns else (df_sorted["f0_GHz"] * 1e9 if "f0_GHz" in df_sorted.columns else pd.Series(dtype=float))
                valid_fr_series = fr_series[(fr_series >= 1e9) & (fr_series <= 100e9)]
                fr_value = valid_fr_series.iloc[0] if len(valid_fr_series) > 0 else np.nan
                
                best_fit_params = None
                best_rss = np.inf
                
                for n_init in N_sat_initials:
                    try:
                        popt, pcov = curve_fit(
                            self._model_func,
                            log_p_filtered,
                            d_filtered,
                            p0=[delta_TLS_init, n_init, alpha_init, delta_0_init],
                            sigma=e_filtered,
                            absolute_sigma=True,
                            maxfev=5000000
                        )
                        
                        fitted_vals = self._model_func(log_p_filtered, *popt)
                        rss = self._calculate_rss(fitted_vals, d_filtered, e_filtered)
                        
                        if rss < best_rss:
                            best_rss = rss
                            best_fit_params = popt
                    except Exception as e:
                        pass
                
                if best_fit_params is not None:
                    delta_TLS, N_sat, alpha, delta_0 = best_fit_params
                    
                    # Apply parameter boundaries and re-fit if necessary
                    if N_sat < self.n_sat_lowerbound:
                        print(f"    N_sat ({N_sat:.2e}) below lowerbound. Refitting with fixed N_sat={self.n_sat_lowerbound}")
                        N_sat = self.n_sat_lowerbound
                        try:
                            popt, pcov = curve_fit(
                                self._model_func_fixed_nsat_low,
                                log_p_filtered,
                                d_filtered,
                                p0=[delta_TLS, alpha, delta_0],
                                sigma=e_filtered,
                                absolute_sigma=True,
                                maxfev=5000000
                            )
                            best_fit_params = np.insert(popt, 1, N_sat)
                        except Exception:
                            pass
                            
                    elif N_sat > self.n_sat_upperbound:
                        print(f"    N_sat ({N_sat:.2e}) above upperbound. Refitting with fixed N_sat={self.n_sat_upperbound}")
                        N_sat = self.n_sat_upperbound
                        try:
                            popt, pcov = curve_fit(
                                self._model_func_fixed_nsat_high,
                                log_p_filtered,
                                d_filtered,
                                p0=[delta_TLS, alpha, delta_0],
                                sigma=e_filtered,
                                absolute_sigma=True,
                                maxfev=5000000
                            )
                            best_fit_params = np.insert(popt, 1, N_sat)
                        except Exception:
                            pass
                            
                    delta_TLS, N_sat, alpha, delta_0 = best_fit_params
                    if alpha < self.alpha_lowerbound:
                        print(f"    alpha ({alpha:.2f}) below lowerbound. Refitting with fixed alpha={self.alpha_lowerbound}")
                        alpha = self.alpha_lowerbound
                        try:
                            popt, pcov = curve_fit(
                                self._model_func_fixed_alpha,
                                log_p_filtered,
                                d_filtered,
                                p0=[delta_TLS, N_sat, delta_0],
                                sigma=e_filtered,
                                absolute_sigma=True,
                                maxfev=5000000
                            )
                            best_fit_params = np.insert(popt, 2, alpha)
                        except Exception:
                            pass
                            
                    delta_TLS, N_sat, alpha, delta_0 = best_fit_params
                    delta_max = np.max(d_filtered)
                    delta_min = np.min(d_filtered)
                    
                    # 1. Output individual fit summary CSV
                    res_tls_entry = {
                        "fr": fr_value,
                        "f0_GHz": fr_value / 1e9 if not np.isnan(fr_value) else np.nan,
                        "delta_TLS": delta_TLS,
                        "N_sat": N_sat,
                        "alpha": alpha,
                        "delta_0": delta_0,
                        "delta_max": delta_max,
                        "delta_min": delta_min
                    }
                    res_tls_df = pd.DataFrame([res_tls_entry])
                    res_tls_csv = os.path.join(resonator_dir, f"{resonator}_tls_fit.csv")
                    res_tls_df.to_csv(res_tls_csv, index=False)
                    print(f"    Saved TLS fit results to: {res_tls_csv}")
                    
                    # Add to global list
                    global_entry = copy.deepcopy(res_tls_entry)
                    global_entry["Resonator"] = resonator
                    global_tls_results.append(global_entry)
                    
                    # 2. Plot TLS loss fit curve
                    plt.figure(figsize=(7, 5))
                    plt.errorbar(p_filtered, d_filtered, yerr=e_filtered, fmt='o', color='#3498db', alpha=0.7, label='Measured Loss')
                    
                    # Generate smooth fit line
                    p_smooth = np.logspace(np.log10(p_filtered.min()) - 0.5, np.log10(p_filtered.max()) + 0.5, 200)
                    d_smooth = self._model_func(np.log10(p_smooth), delta_TLS, N_sat, alpha, delta_0)
                    plt.plot(p_smooth, d_smooth, '-', color='#e74c3c', linewidth=2.5, label='TLS Fit')
                    
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('Intracavity Photon Number $N_{\\mathrm{photons}}$')
                    plt.ylabel('Internal Loss $\\delta_{\\mathrm{tot}} = 1/Q_i$')
                    plt.title(f'TLS Saturation Loss Fit - {resonator} ({fr_value/1e9:.3f} GHz)')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    # Place parameter text box on plot
                    param_text = f"$\\delta_{{TLS}}$: {delta_TLS:.2e}\n$N_{{sat}}$: {N_sat:.2e}\n$\\alpha$: {alpha:.2f}\n$\\delta_0$: {delta_0:.2e}"
                    plt.gca().text(0.05, 0.05, param_text, transform=plt.gca().transAxes,
                                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=10)
                    
                    plt.tight_layout()
                    plot_path = os.path.join(resonator_dir, f"{resonator}_tls_loss_fit.png")
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    print(f"    Saved TLS fit plot to: {plot_path}")
                else:
                    print("    TLS curve fit failed to converge.")
            except Exception as ex:
                print(f"    Error processing TLS analysis: {ex}")
                import traceback
                traceback.print_exc()

        # Save global TLS summary CSV and plot dual-axis graph
        if global_tls_results:
            global_df = pd.DataFrame(global_tls_results)
            cols = ["Resonator"] + [col for col in global_df.columns if col != "Resonator"]
            global_df = global_df[cols]
            global_df = global_df.sort_values(by="fr")
            
            sample_name = res_pd_config.get("sample", {}).get("name", "resonator")
            global_csv_path = os.path.join(base_data_dir, f"{sample_name}_tls_fit_summary.csv")
            global_df.to_csv(global_csv_path, index=False)
            print(f"\nSaved global comprehensive TLS fit summary to: {global_csv_path}")
            print(global_df[["Resonator", "f0_GHz", "delta_TLS", "N_sat", "alpha", "delta_0"]].to_string(index=False))
            
            # Generate Global Dual-axis Plot
            if len(global_df) > 1:
                try:
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    freqs = global_df["f0_GHz"].values
                    delta_tls = global_df["delta_TLS"].values
                    delta_diff = (global_df["delta_max"] - global_df["delta_min"]).values
                    nsats = global_df["N_sat"].values
                    
                    color1 = 'tab:blue'
                    ax1.set_xlabel('Frequency $f_0$ (GHz)', fontsize=12)
                    ax1.set_ylabel('Loss $\\delta_{TLS}$', color=color1, fontsize=12)
                    ax1.plot(freqs, delta_tls, 'o-', color=color1, label='$\\delta_{TLS}$ (Fit)')
                    ax1.plot(freqs, delta_diff, 'd--', color='tab:green', label='$\\delta_{max} - \\delta_{min}$')
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.set_yscale('log')
                    ax1.grid(True, which="both", alpha=0.15)
                    ax1.legend(loc='upper left')
                    
                    ax2 = ax1.twinx()
                    color2 = 'tab:red'
                    ax2.set_ylabel('Saturation Photons $N_{sat}$', color=color2, fontsize=12)
                    ax2.plot(freqs, nsats, 's-', color=color2, label='$N_{sat}$ (Fit)')
                    ax2.tick_params(axis='y', labelcolor=color2)
                    ax2.set_yscale('log')
                    ax2.legend(loc='upper right')
                    
                    plt.title(f'Frequency vs TLS Loss Parameters - {sample_name}', fontsize=14)
                    plt.tight_layout()
                    summary_plot_path = os.path.join(base_data_dir, f"{sample_name}_tls_summary.png")
                    plt.savefig(summary_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved global TLS parameters summary plot to: {summary_plot_path}")
                except Exception as pe:
                    print(f"Failed to generate global TLS plot: {pe}")
