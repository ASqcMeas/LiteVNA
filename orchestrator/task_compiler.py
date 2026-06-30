import os
import copy
import numpy as np
import tomli_w

class TaskCompiler:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def compile_power_tasks(self):
        """
        Reads resonator_PD.toml (via config_manager), calculates adaptive repeat counts,
        and saves power_dep_resonator.toml.
        """
        print("\n" + "="*60)
        print("PHASE 3: COMPILE POWER-DEPENDENT SNR-ADAPTIVE SWEEP LIST")
        print("="*60)
        
        res_pd_config = self.config_manager.res_pd_config
        file_power_task = self.config_manager.file_power_task
        
        if "resonator" not in res_pd_config or not res_pd_config["resonator"]:
            print("Error: No resonators found in resonator_PD.toml. Please run window finding first.")
            return
            
        data_output_folder = res_pd_config.get("output", {}).get("data_path", "data/raw")
        attenuation = res_pd_config["hardware"]["attenuation"]
        
        new_config = {
            "hardware": res_pd_config["hardware"],
            "sample": res_pd_config["sample"],
        }
        measurement = []
        
        for measurement_info in res_pd_config["resonator"]:
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
        
        with open(file_power_task, 'wb') as file:
            tomli_w.dump(new_config, file)
        print(f"Saved compiled sweep task config to: {file_power_task}")
        # Refresh the config manager's configuration files if they are loaded dynamically
        self.config_manager.res_pd_config = self.config_manager.load_toml(self.config_manager.file_res_pd)
