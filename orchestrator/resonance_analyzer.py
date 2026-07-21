import numpy as np
from scipy.signal import find_peaks, peak_widths
from resonator_tools import circuit

class ResonanceAnalyzer:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.vna_config = config_manager.vna_config

    def find_dips(self, freq_array, s_params, expected_count=None, prominence: float | str | None = 2.0, discarded_dips=None):
        """
        Locates downward dips (peaks in -magnitude) and filters/sorts them.
        Filters out dips that do not satisfy the FWHM boundaries.
        Computes FWHM and Depth confidence scores.
        Returns a list of tuples: (freq, magnitude, peak_idx, fwhm, score, s_fwhm, s_depth)
        """
        # Guardrail: ensure freq_array length matches s_params length to prevent IndexError
        if len(freq_array) != len(s_params):
            print(f"  [Guardrail] Length mismatch: freq_array ({len(freq_array)}) vs s_params ({len(s_params)}). Aligning freq_array.")
            if len(freq_array) > 1:
                freq_array = np.linspace(freq_array[0], freq_array[-1], len(s_params))
            else:
                freq_array = np.linspace(0, 1, len(s_params))

        # Use np.maximum to guard against log10(0) = -inf when signal is zero
        magnitude = 20 * np.log10(np.maximum(np.abs(s_params), 1e-18))
        
        # Estimate high-frequency random noise standard deviation
        # Using Median Absolute Deviation (MAD) to be robust against sharp resonator dips (outliers)
        if len(magnitude) > 1:
            diffs = np.diff(magnitude)
            mad = np.median(np.abs(diffs - np.median(diffs)))
            noise_std = 1.4826 * mad / np.sqrt(2)
        else:
            noise_std = 0.05
            
        def parse_optional_float(val, default=None):
            if val is None:
                return default
            if isinstance(val, str) and val.strip().lower() in ["none", "null", "", "auto"]:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        fwhm_config = self.vna_config.get("fwhm", {})
        min_fwhm_raw = parse_optional_float(fwhm_config.get("min_khz", 5.0))
        min_fwhm = min_fwhm_raw * 1e3 if min_fwhm_raw is not None else 5.0 * 1e3

        max_fwhm_raw = parse_optional_float(fwhm_config.get("max_mhz", 30.0))
        max_fwhm = max_fwhm_raw * 1e6 if max_fwhm_raw is not None else 30.0 * 1e6

        target_fwhm = float(fwhm_config.get("target_fwhm_khz", 300.0)) * 1e3
        sigma_dec = float(fwhm_config.get("fwhm_sigma_decade", 0.5))
        ns_mult = float(fwhm_config.get("noise_sigma_multiplier", 6.0))
        min_ns_mult = float(fwhm_config.get("min_noise_sigma_multiplier", 3.0))

        verif_config = self.vna_config.get("verification", {})
        blind_config = self.vna_config.get("blind_search", {})
        min_prom_val = verif_config.get("min_prominence_db", blind_config.get("min_prominence_db", 0.3))
        
        is_min_prom_auto = False
        if min_prom_val is None:
            is_min_prom_auto = True
        elif isinstance(min_prom_val, str) and min_prom_val.strip().lower() == "auto":
            is_min_prom_auto = True
            
        if is_min_prom_auto:
            min_prominence = min_ns_mult * noise_std
        else:
            min_prominence = float(min_prom_val)

        # Calculate standard and floor prominence based on noise (with min_prominence as absolute floor)
        auto_prom = max(min_prominence, ns_mult * noise_std)
        prom_floor = max(min_prominence, min_ns_mult * noise_std)

        print(f"  [Resonance Analyzer] Est. Noise Std: {noise_std:.4f} dB, SNR Threshold: {auto_prom:.2f} dB, SNR Floor: {prom_floor:.2f} dB")
        
        # Save to instance attributes for external logging
        self.last_noise_std = float(noise_std)
        self.last_prominence_floor = float(prom_floor)
        
        # Determine the prominence threshold
        is_auto = False
        if prominence is None:
            is_auto = True
        elif isinstance(prominence, str) and prominence.lower() == "auto":
            is_auto = True
            
        if is_auto:
            start_prom = auto_prom
        else:
            start_prom = float(prominence)
            
        current_prom = start_prom
        peaks, properties = find_peaks(-magnitude, prominence=current_prom)

        # If expected_count is specified, we can dynamically lower the threshold down to prom_floor
        if expected_count is not None:
            target_count = expected_count
            while len(peaks) < target_count and current_prom > prom_floor:
                current_prom -= 0.25
                if current_prom < prom_floor:
                    current_prom = prom_floor
                peaks, properties = find_peaks(-magnitude, prominence=current_prom)
                if current_prom == prom_floor:
                    break
            
        if len(peaks) == 0:
            return []
            
        # Calculate physical FWHM in Hz using linear magnitude depth
        # rel_height = 1 / sqrt(2) (~0.7071) measures down to S21^2 = 0.5 (exact 3dB Lorentzian linewidth)
        lin_mag = np.abs(s_params)
        lin_depth = np.max(lin_mag) - lin_mag
        widths_results = peak_widths(lin_depth, peaks, rel_height=1.0 / np.sqrt(2))
        freq_step = freq_array[1] - freq_array[0] if len(freq_array) > 1 else 1.0
        fwhms = widths_results[0] * freq_step

        res = []
        for orig_idx, p in enumerate(peaks):
            fwhm_val = fwhms[orig_idx]
            
            # Parabolic interpolation to find a more precise center frequency
            precise_freq = freq_array[p]
            if 0 < p < len(freq_array) - 1:
                y1 = magnitude[p - 1]
                y2 = magnitude[p]
                y3 = magnitude[p + 1]
                denom = (y1 - 2.0 * y2 + y3)
                if abs(denom) > 1e-9:
                    d = 0.5 * (y1 - y3) / denom
                    precise_freq = freq_array[p] + d * freq_step

            # Hard Cutoff: FWHM boundaries check (only if configured)
            if min_fwhm_raw is not None and fwhm_val < min_fwhm:
                print(f"  [Filter] Discarding dip at {precise_freq/1e9:.5f} GHz: FWHM too narrow ({fwhm_val/1e3:.1f} kHz < {min_fwhm/1e3:.1f} kHz)")
                if discarded_dips is not None:
                    discarded_dips.append({
                        "freq": precise_freq,
                        "mag": magnitude[p],
                        "fwhm": fwhm_val,
                        "reason": f"FWHM too narrow ({fwhm_val/1e3:.1f} kHz < {min_fwhm/1e3:.1f} kHz)"
                    })
                continue
            if max_fwhm_raw is not None and fwhm_val > max_fwhm:
                print(f"  [Filter] Discarding dip at {precise_freq/1e9:.5f} GHz: FWHM too wide ({fwhm_val/1e6:.1f} MHz > {max_fwhm/1e6:.1f} MHz)")
                if discarded_dips is not None:
                    discarded_dips.append({
                        "freq": precise_freq,
                        "mag": magnitude[p],
                        "fwhm": fwhm_val,
                        "reason": f"FWHM too wide ({fwhm_val/1e6:.1f} MHz > {max_fwhm/1e6:.1f} MHz)"
                    })
                continue
                
            # Log-Gaussian FWHM Score
            log10_fwhm = np.log10(fwhm_val)
            log10_target = np.log10(target_fwhm)
            s_fwhm = 100.0 * np.exp(-((log10_fwhm - log10_target) ** 2) / (2.0 * (sigma_dec ** 2)))
            
            # Linear Depth Score (scaling between 0.5 dB and 10.0 dB)
            peak_prom = properties.get("prominences", [current_prom]*len(peaks))[orig_idx]
            s_depth = min(100.0, max(0.0, (peak_prom - 0.5) / 9.5 * 100.0))
            
            # Read scoring method from configuration
            filtering_config = self.vna_config.get("filtering", {})
            scoring_method = str(filtering_config.get("scoring_method", "geometric")).strip().lower()
            
            if scoring_method == "geometric":
                s_fwhm_safe = max(0.01, s_fwhm)
                s_depth_safe = max(0.01, s_depth)
                initial_score = (s_fwhm_safe ** 0.6) * (s_depth_safe ** 0.4)
            else:
                initial_score = 0.6 * s_fwhm + 0.4 * s_depth
            
            res.append((precise_freq, magnitude[p], p, fwhm_val, initial_score, s_fwhm, s_depth))
            
        # Sort found dips by initial score (highest score first)
        if len(res) > 0:
            res = sorted(res, key=lambda x: x[4], reverse=True)
            
        if expected_count is not None:
            # Return only the top expected_count deepest peaks
            selected_peaks = res[:expected_count]
            # Log the remaining ones as discarded
            if discarded_dips is not None and len(res) > expected_count:
                for dp in res[expected_count:]:
                    discarded_dips.append({
                        "freq": dp[0],
                        "mag": dp[1],
                        "fwhm": dp[3],
                        "score": dp[4],
                        "reason": f"Ranked below expected count limit ({expected_count})"
                    })
            # Re-sort chronologically by frequency index
            selected_peaks = sorted(selected_peaks, key=lambda x: x[2])
            return selected_peaks
            
        return res

    def run_optimized_peak_finding(self, driver, target_fre, deltafre, points, power, IF_bandwidth, expected_dips_count=1):
        """
        Executes Step A (Coarse sweep) and Step B (Fine sweep) with robust retries.
        """
        fwhm_config = self.vna_config.get("fwhm", {})
        min_fwhm = float(fwhm_config.get("min_khz", 50.0)) * 1e3
        max_fwhm = float(fwhm_config.get("max_mhz", 1.0)) * 1e6
        search_window_multiplier = float(fwhm_config.get("search_window_multiplier", fwhm_config.get("window_multiplier", 5.0)))
        fit_measurement_window_multiplier = float(fwhm_config.get("fit_measurement_window_multiplier", fwhm_config.get("window_multiplier", 15.0)))

        retry_config = self.vna_config.get("peak_finding_retry", {})
        max_retries = int(retry_config.get("max_retries", 3))
        retry_1_points = int(retry_config.get("retry_1_points", 1601))
        retry_1_ibw = int(retry_config.get("retry_1_ibw_hz", 200))
        retry_2_mult = float(retry_config.get("retry_2_range_multiplier", 2.0))
        retry_3_boost = float(retry_config.get("retry_3_power_boost_db", 5.0))
        fallback_window = float(retry_config.get("fallback_window_mhz", 2.0)) * 1e6

        current_delta = deltafre
        current_points = points
        current_ibw = IF_bandwidth
        current_power = power
        
        # Step A: Coarse Sweep Retry Loop
        for attempt in range(max_retries):
            start_freq = target_fre - current_delta
            stop_freq = target_fre + current_delta
            
            print(f"  [Step A] Attempt {attempt+1}/{max_retries}: Sweep range {start_freq/1e9:.6f} to {stop_freq/1e9:.6f} GHz...")
            
            freq_array, s_params = driver.measure_sweep(
                start_freq, stop_freq, current_points, 
                driver.vna.current_channel if hasattr(driver.vna, "current_channel") else "S21",
                current_power, current_ibw
            )
            
            # Search for dips
            dips = self.find_dips(freq_array, s_params, expected_count=expected_dips_count)
            
            if dips:
                # Dip successfully found!
                peak_freq, peak_mag, peak_idx, fwhm = dips[0][:4] # Pick the deepest dip
                print(f"  [Step A] Found dip at: {peak_freq/1e9:.6f} GHz ({peak_mag:.2f} dB)")
                print(f"  [Step A] Estimated FWHM: {fwhm/1e6:.3f} MHz")
                
                # Guardrails for FWHM window
                if fwhm < min_fwhm:
                    print(f"  [Guardrail] FWHM {fwhm/1e3:.1f} kHz too small. Capping at {min_fwhm/1e3:.1f} kHz.")
                    fwhm = min_fwhm
                elif fwhm > max_fwhm:
                    print(f"  [Guardrail] FWHM {fwhm/1e6:.2f} MHz too large. Capping at {max_fwhm/1e6:.2f} MHz.")
                    fwhm = max_fwhm
                    
                # Define Step B window (using search_window_multiplier)
                new_start = peak_freq - search_window_multiplier * fwhm
                new_stop = peak_freq + search_window_multiplier * fwhm
                break
            else:
                print("  [Step A] No dip found. Adjusting parameters for retry...")
                if attempt == 0:
                    # Retry 1: Increase resolution and SNR
                    print(f"  -> Retry 1: Increasing points to {retry_1_points}, lowering IF Bandwidth to {retry_1_ibw} Hz.")
                    current_points = retry_1_points
                    current_ibw = retry_1_ibw
                elif attempt == 1:
                    # Retry 2: Widen sweep range
                    print(f"  -> Retry 2: Multiplying search sweep range deltafre by {retry_2_mult}.")
                    current_delta = retry_2_mult * current_delta
                elif attempt == 2:
                    # Retry 3: Adjust power to check saturation or signal level
                    print(f"  -> Retry 3: Boosting VNA power by {retry_3_boost} dB.")
                    current_power = current_power + retry_3_boost
        else:
            # Fallback degradation if no dip is found after all retries
            print(f"  [Degradation] Failed to locate dip around {target_fre/1e9:.5f} GHz after {max_retries} attempts.")
            fallback_half = fallback_window / 2.0
            print(f"  -> Falling back to designed frequency {target_fre/1e9:.5f} GHz with default {fallback_window/1e6:.1f} MHz sweep window.")
            return target_fre - fallback_half, target_fre + fallback_half

        # Step B: Fine Sweep
        print(f"  [Step B] Fine sweep within optimized window: {new_start/1e9:.6f} to {new_stop/1e9:.6f} GHz...")
        freq_array_fine, s_params_fine = driver.measure_sweep(
            new_start, new_stop, points, 
            driver.vna.current_channel if hasattr(driver.vna, "current_channel") else "S21",
            power, IF_bandwidth
        )
        
        dips_fine = self.find_dips(freq_array_fine, s_params_fine, expected_count=expected_dips_count)
        if dips_fine:
            peak_freq, peak_mag, peak_idx, fwhm_fine = dips_fine[0][:4]
            print(f"  [Step B] Found refined dip at: {peak_freq/1e9:.6f} GHz ({peak_mag:.2f} dB)")
            print(f"  [Step B] Refined FWHM: {fwhm_fine/1e6:.3f} MHz")
            
            # Step B Trial Circle Fit Validation (only warning in manual mode)
            try:
                port_str = self.vna_config.get("hardware", {}).get("port", "S21").upper()
                if port_str in ["S11", "S22", "S33", "S44"]:
                    fit_port = circuit.reflection_port(f_data=freq_array_fine, z_data_raw=s_params_fine)
                else:
                    fit_port = circuit.notch_port(f_data=freq_array_fine, z_data_raw=s_params_fine)
                fit_port.autofit()
                if port_str in ["S11", "S22", "S33", "S44"]:
                    qi_val = fit_port.fitresults.get("Qi", np.nan)
                else:
                    qi_val = fit_port.fitresults.get("Qi_dia_corr", np.nan)
                chi_val = fit_port.fitresults.get("chi_square", np.nan)
                
                if np.isnan(qi_val) or qi_val < 0 or np.isnan(chi_val):
                    print(f"  [Step B Verification Fit] Non-physical fit results (Qi={qi_val}, chi={chi_val}). Warning: Potentially false positive.")
                else:
                    print(f"  [Step B Verification Fit] Fit succeeded: Qi={qi_val:.1f}, chi_square={chi_val:.5f}")
            except Exception as fit_err:
                print(f"  [Step B Verification Fit] Trial fit failed: {fit_err}")

            # Apply guardrails
            if fwhm_fine < min_fwhm: fwhm_fine = min_fwhm
            if fwhm_fine > max_fwhm: fwhm_fine = max_fwhm
            
            final_start = peak_freq - fit_measurement_window_multiplier * fwhm_fine
            final_stop = peak_freq + fit_measurement_window_multiplier * fwhm_fine
            return final_start, final_stop
        else:
            print("  [Step B] No dip found in fine sweep. Using Step A optimized window.")
            return new_start, new_stop
