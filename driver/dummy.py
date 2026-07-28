import pyvisa
import numpy as np
import time
from .VNA import VNA

class VNA_DUMMY(VNA):

    def __init__(self, address):
        self.address = address
        print(f'Connected to: {format(address)}')

    @property
    def inst(self) -> str:
        return "dummy"

    def disconnect(self):
        print("VNA_E5080B object connection is closed.")

    def __del__(self):
        print("VNA_E5080B object has been deleted")
        try:
            self.disconnect()
        except AttributeError:
            # In case __inst was not initialized correctly
            print("self.disconnect() failed.")
            pass

    def lin_freq_sweep(self, start, stop, points: int, port: str = "s21", power: float = -20, IF_bandwidth: int = 1000):
        """
        Simulates physical S-parameters containing realistic transmission baseline 
        (overall slope and standing wave ripples) and 9 specific Lorentzian dips 
        matching the DemoTest2 calibration run.
        """
        # 1. Generate frequency array
        freq_array = np.linspace(start, stop, points)
        f_ghz = freq_array / 1e9
        
        # 2. Absolute frequency baseline fit mapped to the [4.0, 8.0] GHz reference range.
        # This ensures that for a 4.0 to 8.0 GHz sweep, the baseline matches the real 
        # calibration run exactly (sloping from +2.3 dB to -11.3 dB).
        # We clip f_ghz to [4.0, 8.0] to prevent the cubic baseline from diverging 
        # when running sweeps outside this range (e.g. 11.0 to 14.0 GHz).
        f_clipped = np.clip(f_ghz, 4.0, 8.0)
        x = f_clipped - 6.0
        dB_base = -2.5 - 2.8 * x - 0.5 * (x ** 2) - 0.15 * (x ** 3)
        
        # 3. Add background ripples (standing waves) using absolute frequency
        ripple_db = 0.6 * np.cos(2 * np.pi * f_ghz / 0.26) + 0.2 * np.sin(2 * np.pi * f_ghz / 0.13)
        dB_total = dB_base + ripple_db
        
        # Convert dB to linear magnitude, and add flat phase (0.0) to ensure circle fit success.
        mag = 10 ** (dB_total / 20.0)
        s21 = mag * np.exp(1j * 0.0)
        
        # 4. Determine resonator frequencies to simulate in this range
        dips_spec = []
        # If the range overlaps with the DemoTest2 range [4.0, 8.0] GHz, include DemoTest2 spec dips.
        # FWHMs are set to ~3-5 MHz so they are resolved with deep dips under larger VNA sweep steps.
        if start <= 8.0e9 and stop >= 4.0e9:
            demo_dips = [
                {"f0": 4.5000e9, "depth": 0.30, "fwhm": 2e7},
                {"f0": 5.8450e9, "depth": 0.60, "fwhm": 1.0e5},
                {"f0": 5.8500e9, "depth": 0.75, "fwhm": 4.0e6},
                {"f0": 6.0000e9, "depth": 0.25, "fwhm": 3.5e6},
                {"f0": 6.1500e9, "depth": 0.95, "fwhm": 1.0e5},
                {"f0": 6.7500e9, "depth": 0.15, "fwhm": 1.0e6},
                {"f0": 7.8000e9, "depth": 0.50, "fwhm": 4.5e6}
            ]
            for d in demo_dips:
                if start <= d["f0"] <= stop:
                    dips_spec.append(d)
        
        # If no dips specified in the range, dynamically generate 5 mock resonators
        # evenly spaced in the middle 80% of the swept band
        if not dips_spec:
            f_min, f_max = start, stop
            fres = list(np.linspace(f_min + 0.1 * (f_max - f_min), 
                                    f_max - 0.1 * (f_max - f_min), 
                                    5))
            for f0 in fres:
                dips_spec.append({"f0": f0, "depth": 0.80, "fwhm": 4.0e6})
                
        # 5. Apply Lorentzian dips
        for dip in dips_spec:
            f0 = dip["f0"]
            depth = dip["depth"]
            fwhm = dip["fwhm"]
            q = f0 / fwhm
            s21 *= (1.0 - depth / (1.0 + 2j * q * (freq_array - f0) / f0))
            
        # 6. Add small random measurement noise (power and bandwidth dependent)
        # Using empirical VNA noise floor model fitted from real measurements.
        # To match realistic VNA physical noise floor exactly, we use noise_multiplier = 1.0.
        noise_multiplier = 10.0
        try:
            import os
            import tomlkit
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vna.toml")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = tomlkit.parse(f.read())
                    if "dummy" in config and "noise_multiplier" in config["dummy"]:
                        noise_multiplier = float(config["dummy"]["noise_multiplier"])
        except Exception:
            pass

        c_base = 2.236e-5 * noise_multiplier
        noise_std = c_base * np.sqrt(IF_bandwidth) * (10 ** (-0.952 * power / 20.0))
        noise = np.random.normal(0, noise_std, points) + 1j * np.random.normal(0, noise_std, points)
        s21 += noise
        
        return freq_array, s21