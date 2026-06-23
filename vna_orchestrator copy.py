import os
import sys
import argparse
import glob
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import tomlkit
import tomli_w
import pandas as pd
from datetime import datetime
from scipy.signal import find_peaks, peak_widths

# Resolve local module paths before importing local packages.
# This ensures imports succeed regardless of the current working directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from resonator_tools import circuit
from driver import get_VNA

# Headless matplotlib backend for automated scripts
matplotlib.use('Agg')

class VNAOrchestrator:
    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = config_dir
        
        # Configuration file paths
        self.file_vna_config = os.path.join(config_dir, "vna.config")
        self.file_res_pd = os.path.join(config_dir, "resonator_PD.toml")
        self.file_power_task = os.path.join(config_dir, "power_dep_resonator.toml")

        # Load configurations
        self.vna_config = self._load_toml(self.file_vna_config)
        self.win_find_config = self.vna_config
        self.res_pd_config = self._load_toml(self.file_res_pd)
        self.meas_lf_config = self.vna_config

    def _load_toml(self, path):
        if not os.path.exists(path):
            print(f"Warning: Configuration file {path} not found.")
            return tomlkit.document()
        with open(path, 'r', encoding='utf-8') as f:
            return tomlkit.parse(f.read())

    def _save_toml(self, config, path):
        # We use tomli_w or standard dump. For preserving comments, tomlkit is used.
        with open(path, 'w', encoding='utf-8') as f:
            f.write(tomlkit.dumps(config))
        print(f"Saved configuration to: {path}")

    def get_vna_connection(self, config):
        address = config["hardware"]["address"]
        model = config["hardware"]["model"]
        # Normalize model name to uppercase for case-insensitive comparison
        model_upper = model.upper() if model else ""
        print(f"Connecting to VNA at {address} (Model: {model})...")
        vna = get_VNA(address, model)
        
        # Check if physical connection was successful (drivers swallow exceptions)
        if model_upper == "ZNB" and vna.__class__.__name__ != "VNA_DUMMY":
            if not hasattr(vna, "vna"):
                raise RuntimeError(f"Failed to connect to ZNB VNA at {address}. Please check if the instrument is powered on and connected to the network.")
        elif model_upper == "E5080B" and vna.__class__.__name__ != "VNA_DUMMY":
            try:
                _ = vna.inst
            except AttributeError:
                raise RuntimeError(f"Failed to connect to E5080B VNA at {address}. Please check if the instrument is powered on and connected to the network.")
                
        if hasattr(vna, "check_error"):
            vna.check_error()
        return vna

    def _get_fres_from_config(self):
        fres = []
        # Try measurement_window_finding.toml
        if "measurement" in self.win_find_config:
            for m in self.win_find_config["measurement"]:
                freq_info = m.get("frequency", {})
                if "resonator_fre" in freq_info:
                    fres.append(float(freq_info["resonator_fre"]))
        
        # Try resonator_PD.toml
        if "resonator" in self.res_pd_config:
            for r in self.res_pd_config["resonator"]:
                freq_info = r.get("frequency", {})
                if "start" in freq_info and "stop" in freq_info:
                    fres.append((float(freq_info["start"]) + float(freq_info["stop"])) / 2.0)
                elif "design_freq" in r:
                    fres.append(float(r["design_freq"]))
                    
        return sorted(list(set(fres)))

    def _generate_dummy_s21(self, freq_array):

        """
        Generates simulated complex S21 data containing Lorentzian dips 
        at specified globally fixed frequencies to allow offline (--dummy) testing.
        """
        # Baseline transmission background with slight slope
        # Guard against divide-by-zero when freq_array has only one point
        freq_span = freq_array[-1] - freq_array[0]
        if freq_span == 0:
            s21 = np.ones(len(freq_array), dtype=complex) * np.exp(1j * 0.1)
        else:
            s21 = (1.0 - 0.05 * (freq_array - freq_array[0]) / freq_span) * np.exp(1j * 0.1)
        
        if not hasattr(self, "dummy_resonator_fres") or not self.dummy_resonator_fres:
            # 1. Try to load from existing configurations
            fres = self._get_fres_from_config()
            if fres:
                self.dummy_resonator_fres = fres
            else:
                # 2. If config has no resonators, generate them dynamically based on the first sweep range
                f_min, f_max = freq_array[0], freq_array[-1]
                # Generate 5 mock resonators evenly spaced in the middle 80% of the swept band
                self.dummy_resonator_fres = list(np.linspace(f_min + 0.1 * (f_max - f_min), 
                                                             f_max - 0.1 * (f_max - f_min), 
                                                             5))
            print(f"[Dummy VNA] Simulated physical resonator frequencies fixed at: "
                  f"{[f/1e9 for f in self.dummy_resonator_fres]} GHz")
            
        for f0 in self.dummy_resonator_fres:
            fwhm = 100e3
            q = f0 / fwhm
            s21 *= (1.0 - 0.8 / (1.0 + 2j * q * (freq_array - f0) / f0))
            
        return s21

    def measure_sweep(self, vna, start, stop, points, port, power, IF_bandwidth):
        """
        Performs a frequency sweep.
        """
        if getattr(vna, "inst", "") == "dummy" or vna.__class__.__name__ == "VNA_DUMMY":
            freq_array = np.linspace(start, stop, points)
            s_params = self._generate_dummy_s21(freq_array)
            # Add small random noise to simulate measurement noise
            s_params += (np.random.normal(0, 1e-4, points) + 1j * np.random.normal(0, 1e-4, points))
            return freq_array, s_params
        else:
            return vna.lin_freq_sweep(start, stop, points, port, power, IF_bandwidth)

    def find_dips(self, freq_array, s_params, expected_count=None, prominence=2.0):
        """
        Locates downward dips (peaks in -magnitude) and filters/sorts them.
        Returns a list of tuples: (freq, magnitude, peak_idx, fwhm)
        """
        # Use np.maximum to guard against log10(0) = -inf when signal is zero
        magnitude = 20 * np.log10(np.maximum(np.abs(s_params), 1e-18))
        
        # Estimate high-frequency random noise standard deviation
        if len(magnitude) > 1:
            diffs = np.diff(magnitude)
            noise_std = np.std(diffs) / np.sqrt(2)
        else:
            noise_std = 0.05
            
        # Define a safety prominence floor based on 6 * noise_std (6-sigma peak finding)
        # to prevent detecting random noise peaks. Minimum floor is 0.3 dB.
        prominence_floor = max(0.3, 6 * noise_std)
        
        # Determine target count to stop stepping down
        # If expected_count is specified, we want to find at least expected_count peaks.
        # If expected_count is None, we want to find at least 1 peak (if starting at the default threshold yields 0).
        target_count = expected_count if expected_count is not None else 1

        
        # Start prominence (start high at 2.5 if expected_count is specified)
        current_prom = max(2.5, prominence) if expected_count is not None else prominence
        
        peaks, properties = find_peaks(-magnitude, prominence=current_prom)
        
        # Step down until we find at least target_count peaks or hit the floor
        while len(peaks) < target_count and current_prom > prominence_floor:
            current_prom -= 0.25
            if current_prom < prominence_floor:
                current_prom = prominence_floor
            peaks, properties = find_peaks(-magnitude, prominence=current_prom)
            if current_prom == prominence_floor:
                break
            
        if len(peaks) == 0:
            return []
            
        # Calculate FWHM in Hz
        widths_results = peak_widths(-magnitude, peaks, rel_height=0.5)
        freq_step = freq_array[1] - freq_array[0] if len(freq_array) > 1 else 1.0
        fwhms = widths_results[0] * freq_step

        # Sort found dips by depth (deepest first)
        sorted_peak_indices = np.argsort(magnitude[peaks])
        sorted_peaks = peaks[sorted_peak_indices]
        
        res = []
        for p in sorted_peaks:
            orig_idx = np.where(peaks == p)[0][0]
            res.append((freq_array[p], magnitude[p], p, fwhms[orig_idx]))
            
        if expected_count is not None:
            # Return only the top expected_count deepest peaks
            selected_peaks = res[:expected_count]
            # Re-sort chronologically by frequency index
            selected_peaks = sorted(selected_peaks, key=lambda x: x[2])
            return selected_peaks
            
        return res

    def run_optimized_peak_finding(self, vna, target_fre, deltafre, points, power, IF_bandwidth, expected_dips_count=1):
        """
        Executes Step A (Coarse sweep) and Step B (Fine sweep) with robust retries.
        """
        fwhm_config = self.vna_config.get("fwhm", {})
        min_fwhm = float(fwhm_config.get("min_khz", 50.0)) * 1e3
        max_fwhm = float(fwhm_config.get("max_mhz", 1.0)) * 1e6
        window_multiplier = float(fwhm_config.get("window_multiplier", 15.0))

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
            
            freq_array, s_params = self.measure_sweep(
                vna, start_freq, stop_freq, current_points, 
                vna.current_channel if hasattr(vna, "current_channel") else "S21",
                current_power, current_ibw
            )
            
            # Search for dips
            dips = self.find_dips(freq_array, s_params, expected_count=expected_dips_count)
            
            if dips:
                # Dip successfully found!
                peak_freq, peak_mag, peak_idx, fwhm = dips[0] # Pick the deepest dip
                print(f"  [Step A] Found dip at: {peak_freq/1e9:.6f} GHz ({peak_mag:.2f} dB)")
                print(f"  [Step A] Estimated FWHM: {fwhm/1e6:.3f} MHz")
                
                # Guardrails for FWHM window
                if fwhm < min_fwhm:
                    print(f"  [Guardrail] FWHM {fwhm/1e3:.1f} kHz too small. Capping at {min_fwhm/1e3:.1f} kHz.")
                    fwhm = min_fwhm
                elif fwhm > max_fwhm:
                    print(f"  [Guardrail] FWHM {fwhm/1e6:.2f} MHz too large. Capping at {max_fwhm/1e6:.2f} MHz.")
                    fwhm = max_fwhm
                    
                # Define Step B window
                new_start = peak_freq - window_multiplier * fwhm
                new_stop = peak_freq + window_multiplier * fwhm
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
        freq_array_fine, s_params_fine = self.measure_sweep(
            vna, new_start, new_stop, points, 
            vna.current_channel if hasattr(vna, "current_channel") else "S21",
            power, IF_bandwidth
        )
        
        dips_fine = self.find_dips(freq_array_fine, s_params_fine, expected_count=expected_dips_count)
        if dips_fine:
            peak_freq, peak_mag, peak_idx, fwhm_fine = dips_fine[0]
            print(f"  [Step B] Found refined dip at: {peak_freq/1e9:.6f} GHz ({peak_mag:.2f} dB)")
            print(f"  [Step B] Refined FWHM: {fwhm_fine/1e6:.3f} MHz")
            
            # Apply guardrails
            if fwhm_fine < min_fwhm: fwhm_fine = min_fwhm
            if fwhm_fine > max_fwhm: fwhm_fine = max_fwhm
            
            final_start = peak_freq - window_multiplier * fwhm_fine
            final_stop = peak_freq + window_multiplier * fwhm_fine
            return final_start, final_stop
        else:
            print("  [Step B] No dip found in fine sweep. Using Step A optimized window.")
            return new_start, new_stop

    def find_all_windows(self, expected_dips_count=1):
        """
        Runs the window finding phase for all resonators defined in measurement_window_finding.toml.
        """
        print("\n" + "="*60)
        print("PHASE 1 & 2: FIND OPTIMIZED FREQUENCY SWEEP WINDOWS")
        print("="*60)
        
        vna = self.get_vna_connection(self.win_find_config)
        measurements = self.win_find_config["measurement"]
        vna_port = self.win_find_config["hardware"]["port"]
        
        refined_resonators = []
        
        try:
            if hasattr(vna, "setup_measurement"):
                vna.setup_measurement(vna_port)
            for idx, task in enumerate(measurements):
                resonator_fre = task["frequency"]["resonator_fre"]
                deltafre = task["frequency"]["deltafre"]
                points = task["frequency"]["points"]
                power = task["power"]
                IF_bandwidth = task["IF_bandwidth"]
                
                print(f"\nResonator {idx + 1}/{len(measurements)} (Design Center: {resonator_fre/1e9:.5f} GHz):")
                final_start, final_stop = self.run_optimized_peak_finding(
                    vna, resonator_fre, deltafre, points, power, IF_bandwidth,
                    expected_dips_count=expected_dips_count
                )
                
                refined_resonators.append({
                    "label": f"C{int(resonator_fre/1e5)}", # e.g. C46090
                    "start": final_start,
                    "stop": final_stop,
                    "design_freq": resonator_fre
                })
        finally:
            print("Disconnecting from VNA...")
            vna.disconnect()
            
        print("\nAll optimized sweep windows successfully located:")
        for r in refined_resonators:
            print(f"  {r['label']}: {r['start']/1e9:.6f} GHz to {r['stop']/1e9:.6f} GHz")
            
        # Dynamically update resonator_PD.toml
        self.update_resonator_pd_config(refined_resonators)

    def update_resonator_pd_config(self, refined_resonators):
        """
        Updates the resonator list in resonator_PD.toml with the found windows.
        """
        print(f"\nUpdating {self.file_res_pd}...")
        
        # Load hardware and sample settings
        self.res_pd_config["hardware"] = self.win_find_config["hardware"]
        self.res_pd_config["sample"] = self.win_find_config["sample"]
        
        # Sort chronologically by design frequency first
        refined_resonators = sorted(refined_resonators, key=lambda x: x["design_freq"])
        
        # Apply Overlap Guard to prevent neighboring resonators from distorting each other's sweeps & fits
        # Fix: compute both left and right neighbor constraints first, then apply the tightest one.
        # The original code modified both 'start' and 'stop' in the left check, which corrupted
        # the subsequent right-side comparison (it saw the already-modified stop value).
        for i in range(len(refined_resonators)):
            f_curr = refined_resonators[i]["design_freq"]
            
            # Compute the maximum allowed half-span imposed by each neighbor
            left_limit = None
            if i > 0:
                f_prev = refined_resonators[i-1]["design_freq"]
                left_limit = (f_curr - f_prev) / 2.0
            
            right_limit = None
            if i < len(refined_resonators) - 1:
                f_next = refined_resonators[i+1]["design_freq"]
                right_limit = (f_next - f_curr) / 2.0
            
            # Apply the most restrictive (smallest) limit from either side at once
            if left_limit is not None or right_limit is not None:
                active_limits = [x for x in [left_limit, right_limit] if x is not None]
                min_half_dist = min(active_limits)
                curr_half_span = max(
                    f_curr - refined_resonators[i]["start"],
                    refined_resonators[i]["stop"] - f_curr
                )
                if curr_half_span > min_half_dist:
                    refined_resonators[i]["start"] = f_curr - min_half_dist
                    refined_resonators[i]["stop"] = f_curr + min_half_dist
                    print(f"  [Overlap Guard] Resonator {refined_resonators[i]['label']}: window exceeds neighbor boundary. Shrinking to symmetric ±{min_half_dist/1e3:.1f} kHz")
        
        # Load defaults from config
        defaults = self.vna_config.get("resonator_defaults", {})
        default_if_bandwidth = int(defaults.get("if_bandwidth", 20))
        default_points = int(defaults.get("points", 501))
        default_max_repeat = int(defaults.get("max_repeat", 5))
        default_min_repeat = int(defaults.get("min_repeat", 1))
        default_power_ave_ratio = float(defaults.get("power_ave_ratio", -140))
        default_power_min = float(defaults.get("power_min", -50))
        default_power_max = float(defaults.get("power_max", 0))
        default_power_step = float(defaults.get("power_step", 5))

        # Build resonator array
        resonators = []
        for r in refined_resonators:
            # Construct entry
            res_entry = tomlkit.table()
            res_entry["label"] = r["label"]
            res_entry["IF_bandwidth"] = default_if_bandwidth
            
            snr = tomlkit.inline_table()
            snr.update({
                "max_repeat": default_max_repeat,
                "min_repeat": default_min_repeat,
                "power_ave_ratio": default_power_ave_ratio
            })
            res_entry["snr"] = snr
            
            pwr = tomlkit.inline_table()
            pwr.update({
                "min": default_power_min,
                "max": default_power_max,
                "step": default_power_step
            })
            res_entry["power"] = pwr
            
            freq = tomlkit.inline_table()
            freq.update({"start": r["start"], "stop": r["stop"], "points": default_points})
            res_entry["frequency"] = freq
            
            resonators.append(res_entry)
            
        self.res_pd_config["resonator"] = resonators
        self._save_toml(self.res_pd_config, self.file_res_pd)

    def blind_search(self, start_freq, stop_freq, expected_count=None, prominence=2.0):
        """
        Task 1: Performs a blind search in [start_freq, stop_freq] to find all active resonators.
        Runs multi-pass sweeps with different power and noise levels, merges candidates,
        verifies ALL candidates first, and then filters based on expected count.
        """
        print("\n" + "="*60)
        print(f"BLIND RESONATOR SEARCH: {start_freq/1e9:.3f} to {stop_freq/1e9:.3f} GHz")
        print("="*60)
        
        # Load configurations from vna.config
        verification_config = self.vna_config.get("verification", {})
        verification_span = float(verification_config.get("window_span_mhz", 10.0)) * 1e6
        verification_points = int(verification_config.get("points", 501))
        verification_power = verification_config.get("power", None)
        if verification_power is not None:
            verification_power = float(verification_power)
        verification_ibw = int(verification_config.get("if_bandwidth_hz", 200))

        dedup_config = self.vna_config.get("deduplication", {})
        coarse_spacing = float(dedup_config.get("coarse_spacing_mhz", 0.25)) * 1e6
        precise_spacing = float(dedup_config.get("precise_spacing_mhz", 0.15)) * 1e6

        fwhm_config = self.vna_config.get("fwhm", {})
        min_fwhm = float(fwhm_config.get("min_khz", 50.0)) * 1e3
        max_fwhm = float(fwhm_config.get("max_mhz", 1.0)) * 1e6
        window_multiplier = float(fwhm_config.get("window_multiplier", 15.0))

        passes_config = self.vna_config.get("blind_search_passes", {}).get("passes", [])
        sweep_passes = []
        for p in passes_config:
            sweep_passes.append({
                "power": float(p.get("power", -20.0)),
                "IF_bandwidth": int(p.get("if_bandwidth", p.get("IF_bandwidth", 1000))),
                "points": int(p.get("points", 16001)),
                "name": str(p.get("name", "Pass"))
            })
        if not sweep_passes:
            sweep_passes = [
                {"power": -15.0, "IF_bandwidth": 1000, "points": 16001, "name": "Pass 1 (High Power, -15 dBm)"},
                {"power": -35.0, "IF_bandwidth": 200, "points": 16001, "name": "Pass 2 (Low Power, -35 dBm)"},
                {"power": -45.0, "IF_bandwidth": 100, "points": 16001, "name": "Pass 3 (Ultra Low Power, -45 dBm)"}
            ]

        vna = self.get_vna_connection(self.win_find_config)
        vna_port = self.win_find_config["hardware"]["port"]
        
        candidate_freqs = []
        cached_sweeps = []
        
        try:
            if hasattr(vna, "setup_measurement"):
                vna.setup_measurement(vna_port)
                
            # Perform multi-pass sweeps
            for p_idx, p_config in enumerate(sweep_passes):
                p_name = p_config["name"]
                p_pow = p_config["power"]
                p_ibw = p_config["IF_bandwidth"]
                p_pts = p_config["points"]
                
                print(f"\n--- Running {p_name} ---")
                freq_array, s_params = self.measure_sweep(
                    vna, start_freq, stop_freq, p_pts, vna_port, p_pow, p_ibw
                )
                
                # Cache results for visualization plot
                cached_sweeps.append((freq_array, s_params))
                
                # Find dips with high prominence to filter out background ripple, but support iterative search if expected_count is specified
                dips = self.find_dips(freq_array, s_params, expected_count=expected_count, prominence=prominence)
                print(f"{p_name} found {len(dips)} dip candidates (prominence >= {prominence} dB).")
                for freq, mag, _, fwhm in dips:
                    # Deduplicate: merge with existing candidates if within adaptive coarse_spacing
                    duplicate_idx = None
                    for c_idx, (f_c, m_c, fwhm_c, p_c) in enumerate(candidate_freqs):
                        spacing = max(coarse_spacing, 0.5 * fwhm, 0.5 * fwhm_c)
                        if abs(freq - f_c) < spacing:
                            duplicate_idx = c_idx
                            break
                            
                    if duplicate_idx is None:
                        print(f"  Candidate (New): {freq/1e9:.5f} GHz ({mag:.2f} dB, coarse FWHM: {fwhm/1e6:.3f} MHz, power: {p_pow} dBm)")
                        candidate_freqs.append((freq, mag, fwhm, p_pow))
                    else:
                        # Keep the deeper one (more negative dB magnitude)
                        f_dup, m_dup, fwhm_dup, p_dup = candidate_freqs[duplicate_idx]
                        if mag < m_dup:
                            print(f"  Candidate {freq/1e9:.5f} GHz ({mag:.2f} dB) replaces duplicate ({m_dup:.2f} dB)")
                            candidate_freqs[duplicate_idx] = (freq, mag, fwhm, p_pow)
                        else:
                            print(f"  Candidate {freq/1e9:.5f} GHz (Skipped, duplicate from previous passes)")
            
            if not candidate_freqs:
                print("No candidate frequencies found in coarse sweeps.")
                return []
                
            print(f"\nTotal candidate frequencies before verification: {len(candidate_freqs)}")
            for idx, (f_c, m_c, fwhm_c, p_c) in enumerate(candidate_freqs):
                print(f"  {idx+1}: {f_c/1e9:.5f} GHz (found at {p_c} dBm)")
                
            # Step B: Verification Sweep (Narrow High-Resolution sweeps around candidates)
            temp_refined_resonators = []
            for idx, (f_c, m_c, fwhm_c, p_c) in enumerate(candidate_freqs):
                print(f"\n--- Verifying Candidate {idx+1}/{len(candidate_freqs)} ({f_c/1e9:.5f} GHz) ---")
                                
                # Dynamic verification span: max(configured_span, 5 * coarse_FWHM)
                span_v = max(verification_span, 5.0 * fwhm_c)
                
                print(f"  [Verification] Adaptive sweep span: {span_v/1e6:.3f} MHz (Range: {(f_c - span_v/2)/1e9:.6f} to {(f_c + span_v/2)/1e9:.6f} GHz)")
                
                v_power = verification_power if verification_power is not None else p_c
                print(f"  [Verification] Power used: {v_power} dBm")
                freq_array_v, s_params_v = self.measure_sweep(
                    vna, f_c - span_v/2, f_c + span_v/2, verification_points, vna_port, v_power, verification_ibw
                )
                
                # Find all dips inside this narrow window to allow resolving very close resonators. Use expected_count=1 to enforce iterative search in verification.
                dips_v = self.find_dips(freq_array_v, s_params_v, expected_count=1, prominence=prominence)
                
                if dips_v:
                    for peak_freq, peak_mag, peak_idx, fwhm_v in dips_v:
                        print(f"  Dip verified at: {peak_freq/1e9:.6f} GHz ({peak_mag:.2f} dB)")
                        print(f"  Calculated FWHM: {fwhm_v/1e6:.3f} MHz")
                        
                        # Apply adaptive guardrails
                        max_fwhm_adaptive = max(max_fwhm, 2.0 * fwhm_c)
                        if fwhm_v < min_fwhm: fwhm_v = min_fwhm
                        if fwhm_v > max_fwhm_adaptive: fwhm_v = max_fwhm_adaptive
                        
                        # Final window using standard FWHM multiple
                        final_start = peak_freq - window_multiplier * fwhm_v
                        final_stop = peak_freq + window_multiplier * fwhm_v
                        
                        temp_refined_resonators.append({
                            "label": f"C{int(peak_freq/1e5)}",
                            "start": final_start,
                            "stop": final_stop,
                            "design_freq": peak_freq,
                            "verified_depth": peak_mag,
                            "fwhm": fwhm_v,
                            "freq_v": freq_array_v,
                            "s21_v": s_params_v
                        })
                else:
                    print(f"  Warning: No dip verified around {f_c/1e9:.5f} GHz. Discarding candidate as false positive.")
            
            # Deduplicate verified resonators based on precise design_freq using adaptive spacing
            unique_refined = []
            for r in temp_refined_resonators:
                duplicate = False
                for ur_idx, ur in enumerate(unique_refined):
                    spacing = max(precise_spacing, 0.5 * r["fwhm"], 0.5 * ur["fwhm"])
                    if abs(r["design_freq"] - ur["design_freq"]) < spacing:
                        duplicate = True
                        # Keep the deeper verified depth (more negative dB value)
                        if r["verified_depth"] < ur["verified_depth"]:
                            unique_refined[ur_idx] = r
                        break
                if not duplicate:
                    unique_refined.append(r)
            temp_refined_resonators = unique_refined
            
            # --- Post-Filtering (Filter-Last) ---
            refined_resonators = []
            if expected_count is not None:
                # Sort verified resonators by verified depth (deepest/lowest dB value first)
                sorted_resonators = sorted(temp_refined_resonators, key=lambda x: x["verified_depth"])
                if len(sorted_resonators) > expected_count:
                    print(f"\nFiltering: Found {len(sorted_resonators)} verified resonators but expected {expected_count}. Selecting the deepest ones.")
                    refined_resonators = sorted_resonators[:expected_count]
                else:
                    refined_resonators = sorted_resonators
                    if len(refined_resonators) < expected_count:
                        print(f"\nWarning: Expected {expected_count} resonators, but only {len(refined_resonators)} verified successfully.")
            else:
                # Keep all verified resonators (dynamic selection)
                print(f"\nDynamic Mode: Keeping all {len(temp_refined_resonators)} successfully verified resonators.")
                refined_resonators = temp_refined_resonators
                
            # Re-sort chronologically by frequency order
            refined_resonators = sorted(refined_resonators, key=lambda x: x["design_freq"])
            
        finally:
            print("Disconnecting from VNA...")
            vna.disconnect()
            
        # Update resonator_PD.toml (this applies Overlap Guard to refined_resonators in-place)
        self.update_resonator_pd_config(refined_resonators)
 
        print("\n=== Blind Search Completed ===")
        print(f"Successfully verified and windowed {len(refined_resonators)} resonators:")
        for r in refined_resonators:
            print(f"  {r['label']}: {r['start']/1e9:.6f} GHz to {r['stop']/1e9:.6f} GHz (verified depth: {r['verified_depth']:.2f} dB)")
            
        # Generate and save visualization plots of the blind search results
        if len(refined_resonators) > 0 and len(cached_sweeps) > 0:
            try:
                base_data_dir = self.res_pd_config.get("output", {}).get("data_path", "data/raw")
                if not os.path.exists(base_data_dir):
                    os.makedirs(base_data_dir)
                    print(f"Created base output directory for plots: {base_data_dir}")
 
                # 1. Save Coarse Sweep Plot (blind_search_coarse.png)
                plt.figure(figsize=(10, 5))
                colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6", "#1abc9c"]
                for p_idx, (freq_c, spar_c) in enumerate(cached_sweeps):
                    mag_c = 20 * np.log10(np.maximum(np.abs(spar_c), 1e-18))
                    p_name = sweep_passes[p_idx]["name"]
                    color = colors[p_idx % len(colors)]
                    plt.plot(freq_c / 1e9, mag_c, color=color, alpha=0.7 - 0.15 * min(p_idx, 2), label=p_name)
                
                # Mark verified resonances
                ylims = plt.ylim()
                for r in refined_resonators:
                    plt.axvline(r["design_freq"] / 1e9, color="green", linestyle="--", alpha=0.7)
                    plt.text(r["design_freq"] / 1e9, ylims[0] + 2, r["label"], color="green", rotation=90, verticalalignment='bottom')
                    
                plt.xlabel("Frequency (GHz)")
                plt.ylabel(f"{vna_port} Magnitude (dB)")
                plt.title("Blind Resonator Search - Coarse Sweeps")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                coarse_path = os.path.join(base_data_dir, "blind_search_coarse.png")
                plt.savefig(coarse_path, dpi=150)
                plt.close()
                print(f"Saved coarse sweep plot to: {coarse_path}")
                
                # 2. Save Combined Fine Sweeps Plot (blind_search_fine_all.png)
                plt.figure(figsize=(10, 5))
                for r in refined_resonators:
                    if "freq_v" in r:
                        # Mask to only draw data within the final fine sweep range [start, stop]
                        mask = (r["freq_v"] >= r["start"]) & (r["freq_v"] <= r["stop"])
                        if np.any(mask):
                            mag_v = 20 * np.log10(np.maximum(np.abs(r["s21_v"]), 1e-18))
                            plt.plot((r["freq_v"][mask] - r["design_freq"]) / 1e6, mag_v[mask], 
                                     label=f"{r['label']} ({r['design_freq']/1e9:.5f} GHz)")
                        
                plt.xlabel("Frequency Offset from Center (MHz)")
                plt.ylabel(f"{vna_port} Magnitude (dB)")
                plt.title("Verified Resonators - Combined Fine Sweeps (Final Range)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fine_all_path = os.path.join(base_data_dir, "blind_search_fine_all.png")
                plt.savefig(fine_all_path, dpi=150)
                plt.close()
                print(f"Saved combined fine sweeps plot to: {fine_all_path}")
                
                # 3. Save N individual fine sweeps (blind_search_fine_<label>.png)
                for r in refined_resonators:
                    if "freq_v" in r:
                        # Mask to only draw data within the final fine sweep range [start, stop]
                        mask = (r["freq_v"] >= r["start"]) & (r["freq_v"] <= r["stop"])
                        if np.any(mask):
                            plt.figure(figsize=(7, 4.5))
                            mag_v = 20 * np.log10(np.maximum(np.abs(r["s21_v"]), 1e-18))
                            plt.plot(r["freq_v"][mask] / 1e9, mag_v[mask], '-', color="#3498db", linewidth=1.5, label="rawdata")
                            plt.axvline(r["design_freq"] / 1e9, color="#e74c3c", linestyle="--", 
                                        label=f"Center ({r['design_freq']/1e9:.5f} GHz)")
                            plt.xlim(r["start"] / 1e9, r["stop"] / 1e9)
                            plt.xlabel("Frequency (GHz)")
                            plt.ylabel(f"{vna_port} Magnitude (dB)")
                            plt.title(f"Verification Fine Sweep - Resonator {r['label']} (Final Range)")
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            plt.tight_layout()
                            indiv_path = os.path.join(base_data_dir, f"blind_search_fine_{r['label']}.png")
                            plt.savefig(indiv_path, dpi=120)
                            plt.close()
                            print(f"Saved individual fine sweep plot to: {indiv_path}")
            except Exception as e:
                print(f"Warning: Failed to generate visualization plots: {e}")

    def compile_power_tasks(self):
        """
        Reads resonator_PD.toml, calculates adaptive repeat counts, and saves power_dep_resonator.toml.
        """
        print("\n" + "="*60)
        print("PHASE 3: COMPILE POWER-DEPENDENT SNR-ADAPTIVE SWEEP LIST")
        print("="*60)
        
        if "resonator" not in self.res_pd_config or not self.res_pd_config["resonator"]:
            print("Error: No resonators found in resonator_PD.toml. Please run window finding first.")
            return
            
        data_output_folder = self.res_pd_config.get("output", {}).get("data_path", "data/raw")
        attenuation = self.res_pd_config["hardware"]["attenuation"]
        
        new_config = {
            "hardware": self.res_pd_config["hardware"],
            "sample": self.res_pd_config["sample"],
        }
        measurement = []
        
        for measurement_info in self.res_pd_config["resonator"]:
            label = measurement_info["label"]
            single_resonator = {
                "label": label,
                "frequency": measurement_info["frequency"],
                "IF_bandwidth": measurement_info["IF_bandwidth"]
            }
            
            vna_power_setting = measurement_info["power"]
            if "custom" not in vna_power_setting.keys():
                min_power = vna_power_setting["min"] 
                max_power = vna_power_setting["max"] 
                step = vna_power_setting["step"] 
                power_range = np.arange(min_power, max_power + step/2, step)
            else:
                power_range = np.array(vna_power_setting["custom"])
                
            max_repeat = int(measurement_info["snr"]["max_repeat"])
            min_repeat = int(measurement_info["snr"]["min_repeat"])
            
            print(f"Compiling power sweeps for {label}...")
            for p in power_range:
                r_sn = measurement_info["snr"]["power_ave_ratio"]
                repeat = 10**((r_sn - (p - attenuation)) / 10)
                if repeat < min_repeat: repeat = min_repeat
                if repeat > max_repeat: repeat = max_repeat
                
                single_power = copy.deepcopy(single_resonator)
                single_power["power"] = float(p)
                single_power["repeat"] = int(np.round(repeat))
                # Use fixed-point format to avoid floating-point precision issues in path names
                # e.g. np.arange can produce -19.999999... instead of -20.0
                single_power["output"] = f"{data_output_folder}/{label}/att{attenuation}_{float(p):.1f}"
                
                measurement.append(single_power)
                
        new_config["measurement"] = measurement
        
        with open(self.file_power_task, 'wb') as file:
            tomli_w.dump(new_config, file)
        print(f"Saved compiled sweep task config to: {self.file_power_task}")

    def run_power_sweep(self):
        """
        Executes the compiled power sweep tasks list, storing NC files.
        """
        print("\n" + "="*60)
        print("PHASE 4: EXECUTE POWER SWEEP")
        print("="*60)
        
        if not os.path.exists(self.file_power_task):
            print(f"Error: Compiled sweep file {self.file_power_task} not found. Please compile tasks first.")
            return
            
        with open(self.file_power_task, 'r', encoding='utf-8') as f:
            sweep_config = tomlkit.parse(f.read())
            
        vna_address = sweep_config["hardware"]["address"]
        vna_model = sweep_config["hardware"]["model"]
        vna_port = sweep_config["hardware"]["port"]
        attenuation = sweep_config["hardware"]["attenuation"]
        measurements = sweep_config["measurement"]
        
        # Use get_vna_connection to apply the same connection validation logic
        # (model name normalization, connection check) used in other phases
        vna = self.get_vna_connection(sweep_config)
        
        try:
            if hasattr(vna, "setup_measurement"):
                vna.setup_measurement(vna_port)
            for m_idx, m_task in enumerate(measurements):
                label = m_task["label"]
                output_folder = m_task["output"]
                freq_start = m_task["frequency"]["start"]
                freq_stop = m_task["frequency"]["stop"]
                sweep_point = m_task["frequency"]["points"]
                vna_power = m_task["power"]
                IF_bandwidth = m_task["IF_bandwidth"]
                repeat = m_task.get("repeat", 1)
                
                print(f"\nTask {m_idx + 1}/{len(measurements)}: Resonator {label} at {vna_power} dBm (repeat={repeat})...")
                
                for i in range(repeat):
                    start_time = datetime.now()
                    
                    # Run sweep
                    freq_array, s_params = self.measure_sweep(
                        vna, freq_start, freq_stop, sweep_point, vna_port, vna_power, IF_bandwidth
                    )
                    
                    end_time = datetime.now()
                    
                    # Save NC file
                    vna_port_str = str(vna_port)
                    var_key = vna_port_str.lower()
                    output_data = {
                        var_key: (["s_params", "frequency"], np.array([s_params.real, s_params.imag]))
                    }
                    dataset = xr.Dataset(
                        output_data,
                        coords={"s_params": np.array(["real", "imag"]), "frequency": freq_array}
                    )
                    dataset.attrs["IF_bandwidth"] = int(IF_bandwidth)
                    dataset.attrs["power"] = float(vna_power)
                    dataset.attrs["attenuation"] = int(attenuation)
                    dataset.attrs["port"] = str(vna_port_str)
                    dataset.attrs["start_time"] = str(start_time.strftime("%Y%m%d_%H%M%S"))
                    dataset.attrs["end_time"] = str(end_time.strftime("%Y%m%d_%H%M%S"))
                    
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    
                    # Include loop index suffix to prevent filename collision when repeat > 1
                    # and a sweep completes within the same second (common in dummy/fast-sweep mode)
                    file_path = f"{output_folder}/{label}_{start_time.strftime('%Y%m%d_%H%M%S')}_{i+1:02d}.nc"
                    dataset.to_netcdf(file_path)
                    print(f"  [Sweep {i+1}/{repeat}] Saved dataset: {file_path}")
        finally:
            print("Disconnecting from VNA...")
            vna.disconnect()

    def run_batch_fitting(self):
        """
        Fits all measured resonators and plots curves, exporting summary CSVs.
        """
        print("\n" + "="*60)
        print("PHASE 5: RUN BATCH RESONANCE FITTING")
        print("="*60)
        
        base_data_dir = self.res_pd_config.get("output", {}).get("data_path", "data/raw")
        if not os.path.exists(base_data_dir):
            print(f"Error: Base output directory {base_data_dir} does not exist. No data to fit.")
            return
            
        resonators = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
        print(f"Found resonators to fit: {resonators}")
        
        global_fit_data_list = []
        
        for resonator in sorted(resonators):
            resonator_dir = os.path.join(base_data_dir, resonator)
            nc_files = sorted(glob.glob(os.path.join(resonator_dir, "**", "*.nc"), recursive=True))
            if not nc_files:
                continue
                
            print(f"\nFitting resonator: {resonator} ({len(nc_files)} files)")
            fit_data_list = []
            
            for file_path in nc_files:
                try:
                    # Use context manager to ensure the NetCDF file handle is closed after
                    # reading, preventing file descriptor exhaustion with many .nc files
                    with xr.open_dataset(file_path) as ds:
                        freq = ds.frequency.values
                        
                        # Dynamically look for the correct variable key (e.g. s21, s11)
                        # Support legacy files with "s21" or dynamic keys named after standard S-parameters
                        s_param_keys = ["s21", "s11", "s22", "s12", "s33", "s44", "s13", "s31", "s14", "s41", "s23", "s32", "s24", "s42", "s34", "s43"]
                        var_key = "s21" # Default fallback
                        for k in ds.data_vars:
                            if k.lower() in s_param_keys:
                                var_key = k
                                break
                        if var_key not in ds.data_vars and len(ds.data_vars) > 0:
                            var_key = list(ds.data_vars.keys())[0]
                            
                        s_param_data = ds[var_key].values
                        complex_s = s_param_data[0] + 1j * s_param_data[1]
                        
                        power = ds.attrs.get("power", np.nan)
                        attenuation = ds.attrs.get("attenuation", 0)
                        p_sample = power - attenuation
                        
                        # Determine S-parameter port from NC attributes, or var_key, or config fallback
                        port_str = ds.attrs.get("port", var_key).upper()
                    # ds is now closed; freq, complex_s, power, attenuation, port_str are plain Python/NumPy values
                    
                    # Select fitting model based on port type
                    # Reflection parameters (S11, S22, S33, S44) use reflection_port
                    if port_str in ["S11", "S22", "S33", "S44"]:
                        print(f"  [Fit] Using Reflection model (reflection_port) for {port_str}...")
                        port = circuit.reflection_port(f_data=freq, z_data_raw=complex_s)
                    else:
                        print(f"  [Fit] Using Transmission model (notch_port) for {port_str}...")
                        port = circuit.notch_port(f_data=freq, z_data_raw=complex_s)
                        
                    port.autofit()
                    
                    # Dynamically center and scale the plotting X-axis around the fitted resonance frequency fr
                    fr_fit = port.fitresults.get("fr", np.nan)
                    ql_fit = port.fitresults.get("Ql", np.nan)
                    
                    # 1. IQ Plot (Re vs Im)
                    plt.figure(figsize=(6, 5))
                    plt.plot(port.z_data_raw.real, port.z_data_raw.imag, '.', label="rawdata", color="#3498db", alpha=0.6)
                    plt.plot(port.z_data_sim.real, port.z_data_sim.imag, '-', label="fit", color="#e74c3c", linewidth=2)
                    plt.xlabel("Re(S)")
                    plt.ylabel("Im(S)")
                    plt.title(f"{resonator} at {power} dBm - Resonance Circle")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(file_path.replace(".nc", "_fit_IQ.png"), dpi=120)
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
                    plt.plot(port.f_data * 1e-9, np.absolute(port.z_data_raw), '.', label="rawdata", color="#3498db", alpha=0.6)
                    plt.plot(port.f_data * 1e-9, np.absolute(port.z_data_sim), '-', label="fit", color="#e74c3c", linewidth=2)
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("|S| Magnitude")
                    plt.xlim(xlim_range)
                    plt.title(f"{resonator} at {power} dBm - Amplitude")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(file_path.replace(".nc", "_fit_amplitude.png"), dpi=120)
                    plt.close()
                    
                    # 3. Phase Plot (arg(S) vs Frequency)
                    plt.figure(figsize=(6, 4))
                    plt.plot(port.f_data * 1e-9, np.angle(port.z_data_raw), '.', label="rawdata", color="#3498db", alpha=0.6)
                    plt.plot(port.f_data * 1e-9, np.angle(port.z_data_sim), '-', label="fit", color="#e74c3c", linewidth=2)
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Phase (rad)")
                    plt.xlim(xlim_range)
                    plt.title(f"{resonator} at {power} dBm - Phase")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(file_path.replace(".nc", "_fit_phase.png"), dpi=120)
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
                        "File": os.path.basename(file_path),
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
                    
                    print(f"  Fit succeeded: {os.path.basename(file_path)} at {power} dBm (Qi: {qi_val:.1f})")
                except Exception as e:
                    print(f"  Fit failed for {os.path.basename(file_path)}: {e}")
                    
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
            sample_name = self.res_pd_config.get("sample", {}).get("name", "resonator")
            global_csv_path = os.path.join(base_data_dir, f"{sample_name}_global_fit_summary.csv")
            global_df.to_csv(global_csv_path, index=False)
            print(f"\nSaved global comprehensive fit summary to: {global_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified VNA Resonator Measurement Orchestrator")
    parser.add_argument("--find-windows", action="store_true", help="Run optimized peak finding and windowing (Phase 1 & 2)")
    parser.add_argument("--compile-tasks", action="store_true", help="Compile power-dependent SNR-adaptive task list (Phase 3)")
    parser.add_argument("--run-sweep", action="store_true", help="Execute the VNA power sweep (Phase 4)")
    parser.add_argument("--fit", action="store_true", help="Perform batch circle fitting on measured data (Phase 5)")
    parser.add_argument("--run-all", action="store_true", help="Run full orchestrator pipeline (Phase 1 to 5)")
    parser.add_argument("--expected-dips", type=int, default=None, help="Expected number of dips to find in the sweep range")
    parser.add_argument("--dummy", action="store_true", help="Force VNA model to DUMMY for offline testing")
    parser.add_argument("--config-dir", type=str, default=None, help="Directory containing configuration TOML files")
    
    # Blind search options
    parser.add_argument("--blind-search", action="store_true", help="Perform a blind search for resonators across a frequency range")
    parser.add_argument("--start-freq", type=float, default=4.0, help="Blind search start frequency in GHz (default: 4.0)")
    parser.add_argument("--stop-freq", type=float, default=8.0, help="Blind search stop frequency in GHz (default: 8.0)")
    parser.add_argument("--prominence", type=float, default=2.0, help="Prominence threshold in dB for finding dips (default: 2.0)")
    parser.add_argument("--sample-name", type=str, default=None, help="Automatically override the sample name in configurations")
    parser.add_argument("--vna-ip", type=str, default=None, help="Automatically override the VNA IP address or VISA address in configurations")
    parser.add_argument("--port", type=str, default=None, help="Override the measured S-parameter port (e.g. S21, S11, S22... S44) in configurations")
    
    args = parser.parse_args()
    
    # Initialize Orchestrator
    orchestrator = VNAOrchestrator(args.config_dir)
    
    # Load defaults from execution/blind_search section of vna.config if not provided via CLI
    exec_config = orchestrator.vna_config.get("execution", {})
    
    # Dummy mode override
    if not args.dummy and exec_config.get("dummy", False):
        args.dummy = True
        
    # Sample name override (checks execution.sample_name first, then falls back to sample.name)
    if not args.sample_name:
        args.sample_name = exec_config.get("sample_name", orchestrator.vna_config.get("sample", {}).get("name", None))
        
    # Expected dips override
    if args.expected_dips is None:
        args.expected_dips = exec_config.get("expected_dips", None)
        
    # Blind search default overrides
    blind_config = orchestrator.vna_config.get("blind_search", {})
    if args.start_freq == 4.0 and "start_freq_ghz" in blind_config:
        args.start_freq = float(blind_config["start_freq_ghz"])
    if args.stop_freq == 8.0 and "stop_freq_ghz" in blind_config:
        args.stop_freq = float(blind_config["stop_freq_ghz"])
    if args.prominence == 2.0 and "prominence_db" in blind_config:
        args.prominence = float(blind_config["prominence_db"])
        
    # Action override
    action = exec_config.get("action", None)
    if action and not (args.find_windows or args.compile_tasks or args.run_sweep or args.fit or args.run_all or args.blind_search):
        if action == "run-all":
            args.run_all = True
        elif action == "blind-search":
            args.blind_search = True
        elif action == "find-windows":
            args.find_windows = True
        elif action == "compile":
            args.compile_tasks = True
        elif action == "sweep":
            args.run_sweep = True
        elif action == "fit":
            args.fit = True

    # If still no action is specified, default to running the full pipeline (--run-all) with blind search
    if not (args.find_windows or args.compile_tasks or args.run_sweep or args.fit or args.run_all or args.blind_search):
        print("No action specified in CLI or config. Defaulting to running the full pipeline (--run-all) with blind search.")
        args.run_all = True
    
    # Determine sample name (use override, existing non-default, or generate a timestamped default)
    sample_name = args.sample_name
    if not sample_name:
        current_name = orchestrator.res_pd_config.get("sample", {}).get("name", "")
        # If the sample name is empty, default, or a mock placeholder, generate a timestamped default
        if not current_name or current_name in ["resonator", "resonator0622", "CLI_Test_Sample"]:
            sample_name = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"No sample name specified (or default detected). Automatically generated sample name: {sample_name}")
        else:
            sample_name = current_name
            
    if sample_name:
        print(f"Overriding sample name in configurations to: {sample_name}")
        if "sample" in orchestrator.vna_config:
            orchestrator.vna_config["sample"]["name"] = sample_name
        if "sample" in orchestrator.res_pd_config:
            orchestrator.res_pd_config["sample"]["name"] = sample_name
            
    # Set dynamic data path only if we are starting a new measurement/search workflow
    # (prevents overriding the data path during standalone fits or sweeps of existing directories)
    generate_new_timestamp = args.run_all or args.blind_search or args.find_windows
    if generate_new_timestamp:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dynamic_data_path = f"data/{sample_name}_{run_timestamp}"
        print(f"Dynamically setting base output data path to: {dynamic_data_path}")
        if "output" not in orchestrator.res_pd_config:
            orchestrator.res_pd_config["output"] = tomlkit.table()
        orchestrator.res_pd_config["output"]["data_path"] = dynamic_data_path
        
    # Physically save configs
    orchestrator._save_toml(orchestrator.vna_config, orchestrator.file_vna_config)
    orchestrator._save_toml(orchestrator.res_pd_config, orchestrator.file_res_pd)
        
    if args.vna_ip:
        visa_address = args.vna_ip
        if "::" not in visa_address:
            visa_address = f"TCPIP0::{visa_address}::inst0::INSTR"
        print(f"Overriding VNA address to: {visa_address}")
        if "hardware" in orchestrator.vna_config:
            orchestrator.vna_config["hardware"]["address"] = visa_address
        if "hardware" in orchestrator.res_pd_config:
            orchestrator.res_pd_config["hardware"]["address"] = visa_address
        # Physically save configs
        orchestrator._save_toml(orchestrator.vna_config, orchestrator.file_vna_config)
        orchestrator._save_toml(orchestrator.res_pd_config, orchestrator.file_res_pd)
        
    if args.port:
        port_val = args.port.upper()
        import re
        if not re.match(r'^S[1-4][1-4]$', port_val):
            print(f"Warning: Specified port '{args.port}' does not match standard S-parameter format (e.g. S11, S21, S22... S44). Proceeding anyway.")
        print(f"Overriding VNA port in configurations to: {port_val}")
        if "hardware" in orchestrator.vna_config:
            orchestrator.vna_config["hardware"]["port"] = port_val
        if "hardware" in orchestrator.res_pd_config:
            orchestrator.res_pd_config["hardware"]["port"] = port_val
        # Physically save configs
        orchestrator._save_toml(orchestrator.vna_config, orchestrator.file_vna_config)
        orchestrator._save_toml(orchestrator.res_pd_config, orchestrator.file_res_pd)
        
    # Override settings for dummy offline test if requested (keeps it local to execution)
    if args.dummy:
        print("[Offline mode] Overriding VNA Model to DUMMY...")
        if "hardware" not in orchestrator.vna_config:
            orchestrator.vna_config["hardware"] = tomlkit.table()
        if "hardware" not in orchestrator.res_pd_config:
            orchestrator.res_pd_config["hardware"] = tomlkit.table()
        orchestrator.vna_config["hardware"]["model"] = "DUMMY"
        orchestrator.vna_config["hardware"]["address"] = "DUMMY_VISA_ADDRESS"
        orchestrator.res_pd_config["hardware"]["model"] = "DUMMY"
        orchestrator.res_pd_config["hardware"]["address"] = "DUMMY_VISA_ADDRESS"
        
    # Note: Default action handling was moved to the top of main()
        
    if args.run_all or args.blind_search:
        # Run blind search
        orchestrator.blind_search(args.start_freq * 1e9, args.stop_freq * 1e9, expected_count=args.expected_dips, prominence=args.prominence)
    elif args.find_windows:
        # Run manual legacy window finding from TOML
        orchestrator.find_all_windows(expected_dips_count=args.expected_dips)
        
    if args.compile_tasks or args.run_all:
        orchestrator.compile_power_tasks()
        
    if args.run_sweep or args.run_all:
        orchestrator.run_power_sweep()
        
    if args.fit or args.run_all:
        orchestrator.run_batch_fitting()

if __name__ == "__main__":
    main()
