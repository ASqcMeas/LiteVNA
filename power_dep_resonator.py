import os
import sys
import numpy as np
import tomlkit
import tomli_w
import copy

# Resolve absolute paths relative to script location
config_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(config_dir, 'resonator_PD.toml')
output_path = os.path.join(config_dir, 'power_dep_resonator.toml')

if not os.path.exists(config_path):
    print(f"Error: {config_path} not found.")
    sys.exit(1)

print(f"Loading {config_path}...")
with open(config_path, 'r', encoding='utf-8') as file:
    content = file.read()
    resPD_config = tomlkit.parse(content)

data_output_folder = resPD_config["output"]["data_path"]

new_config = {
    "hardware": resPD_config["hardware"],
    "sample": resPD_config["sample"],
}
measurement = []

attenuation = resPD_config["hardware"]["attenuation"]
for measurement_info in resPD_config["resonator"]:
    single_resonator = {
        "label": measurement_info["label"],
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

    print(f"Compiling tasks for resonator {measurement_info['label']}...")
    for p in power_range:
        r_sn = measurement_info["snr"]["power_ave_ratio"]
        repeat = 10**((r_sn - (p - attenuation)) / 10)
        if repeat < min_repeat: repeat = min_repeat
        if repeat > max_repeat: repeat = max_repeat

        single_power = copy.deepcopy(single_resonator)
        single_power["power"] = float(p)
        single_power["repeat"] = int(np.round(repeat))
        single_power["output"] = f"{data_output_folder}/{measurement_info['label']}/att{attenuation}_{str(p)}"

        measurement.append(single_power)

new_config["measurement"] = measurement

print(f"Saving compiled measurement tasks to {output_path}...")
with open(output_path, 'wb') as file:
    tomli_w.dump(new_config, file)

print("Done! Config generated successfully.")
