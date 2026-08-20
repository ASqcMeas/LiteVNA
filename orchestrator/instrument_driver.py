from driver import get_VNA

class InstrumentDriver:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.vna = None
        self.dummy_resonator_fres = None

    def get_vna_connection(self, config):
        address = config["hardware"]["address"]
        model = config["hardware"]["model"]
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

    def connect(self):
        self.vna = self.get_vna_connection(self.config_manager.win_find_config)
        return self.vna

    def disconnect(self):
        if self.vna is not None:
            print("Disconnecting from VNA...")
            try:
                self.vna.disconnect()
            except Exception as e:
                print(f"Error disconnecting: {e}")
            self.vna = None

    def setup_measurement(self, port):
        if self.vna is not None and hasattr(self.vna, "setup_measurement"):
            self.vna.setup_measurement(port)

    def measure_sweep(self, start, stop, points, port, power, IF_bandwidth) -> tuple:
        """
        Performs a frequency sweep.
        """
        if self.vna is None:
            raise RuntimeError("VNA is not connected. Call connect() first.")
        return self.vna.lin_freq_sweep(start, stop, points, port, power, IF_bandwidth)

