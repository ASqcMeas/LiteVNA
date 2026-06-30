import os
import tomlkit

class ConfigManager:
    def __init__(self, config_dir=None):
        if config_dir is None:
            # Fallback to current directory of the caller
            config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = config_dir
        
        # Configuration file paths
        self.file_vna_config = os.path.join(config_dir, "vna.toml")
        self.file_res_pd = os.path.join(config_dir, "resonator_PD.toml")
        self.file_power_task = os.path.join(config_dir, "power_dep_resonator.toml")

        # Load configurations
        self.vna_config = self.load_toml(self.file_vna_config)
        self.win_find_config = self.vna_config
        self.res_pd_config = self.load_toml(self.file_res_pd)
        self.meas_lf_config = self.vna_config

    def load_toml(self, path):
        if not os.path.exists(path):
            print(f"Warning: Configuration file {path} not found.")
            return tomlkit.document()
        with open(path, 'r', encoding='utf-8') as f:
            return tomlkit.parse(f.read())

    def save_toml(self, config, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(tomlkit.dumps(config))
        print(f"Saved configuration to: {path}")

    def get_fres_from_config(self):
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
        self.save_toml(self.res_pd_config, self.file_res_pd)
