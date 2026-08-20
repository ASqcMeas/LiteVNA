import numpy as np
import matplotlib.pyplot as plt
from qcodes.instrument_drivers.rohde_schwarz import (
    RohdeSchwarzZNB20,
    RohdeSchwarzZNBChannel,
)
from qcodes.instrument.instrument import Instrument

class VNA_ZNB20:
    def __init__(self, address: str):
        self.address = address
        self.connect()

    def connect(self) -> bool:
        import time
        
        # Explicitly close any existing instrument named 'VNA' in QCodes registry to avoid duplicate errors
        try:
            if 'VNA' in Instrument._all_instruments:
                Instrument._all_instruments['VNA'].close()
        except Exception as e:
            print(f"Error closing existing QCodes 'VNA' instrument: {e}")

        max_connect_attempts = 3
        for attempt in range(1, max_connect_attempts + 1):
            try:
                self.vna = RohdeSchwarzZNB20('VNA', self.address, timeout=300)
                print(f'Connected to: {self.vna.IDN()}')
                return True
            except Exception as e:
                print(f"Connection attempt {attempt}/{max_connect_attempts} to VNA at {self.address} failed: {e}")
                if attempt < max_connect_attempts:
                    time.sleep(2)
        return False

    def reconnect(self) -> bool:
        print(f"Attempting to reconnect to VNA at {self.address}...")
        self.disconnect()
        return self.connect()
    
    def check_error(self):
        # Check for errors
        pass

    def delete_all_traces(self):
        try:
            self.vna.clear_channels()
        except Exception as e:
            print(f"Error clearing channels: {e}. Retrying after reconnect...")
            self.reconnect()
            self.vna.clear_channels()
    
    def setup_measurement(self, parameter: str):
        try:
            self.vna.add_channel(parameter)
            self.current_channel = parameter
        except Exception as e:
            print(f"Error setting up measurement channel '{parameter}': {e}. Retrying after reconnect...")
            self.reconnect()
            self.vna.add_channel(parameter)
            self.current_channel = parameter


    def set_linfreq(self, start: float, stop: float):
        getattr(self.vna.channels, self.current_channel).start(start)
        getattr(self.vna.channels, self.current_channel).stop(stop)

    def set_data_format(self):
        self.vna.write('FORMAT:DATA REAL,64')

    def set_sweep_points(self, points: int):
        getattr(self.vna.channels, self.current_channel).npts(points)

    def set_IF_bandwidth(self, bandwidth: int):
        getattr(self.vna.channels, self.current_channel).bandwidth(bandwidth)

    def get_data(self):
        magnitude_phase_data = getattr(self.vna.channels, self.current_channel).trace_db_phase.get()
        magnitudes_db = magnitude_phase_data[0]
        phases = magnitude_phase_data[1]
        magnitudes_linear = 10**(magnitudes_db/20)

        real_parts = magnitudes_linear * np.cos(phases)
        imaginary_parts = magnitudes_linear * np.sin(phases)
        data = real_parts + 1j*imaginary_parts
        return data

    def set_power(self, power: float):
        getattr(self.vna.channels, self.current_channel).power(power)

    def measure(self):
        self.vna.rf_on()
        self.vna.channels.avg(1)
        # Disable continuous sweep, trigger a single sweep, and wait for completion
        # before reading data. This mirrors the *OPC? synchronization in E5080B.py
        # and prevents returning stale data from a previous sweep's buffer.
        self.vna.write(':INIT:CONT OFF')  # Stop continuous sweep mode
        self.vna.write(':INIT:IMM')        # Trigger one sweep
        self.vna.ask('*OPC?')              # Block until sweep is complete

    def lin_freq_sweep(self, start, stop, points: int, port, power: float = -20, IF_bandwith: int = 1000):
        import time
        import pyvisa

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    print(f"Retrying VNA sweep (attempt {attempt}/{max_attempts})...")
                    self.reconnect()
                
                self.delete_all_traces()
                self.setup_measurement(port)
                self.set_power(power)
                self.set_IF_bandwidth(IF_bandwith)
                self.set_linfreq(start, stop)
                self.set_sweep_points(points)

                self.measure()
                data = self.get_data()

                start_freq = getattr(self.vna.channels, self.current_channel).start()
                stop_freq = getattr(self.vna.channels, self.current_channel).stop()
                s21_data = data
                num_points = len(s21_data) if s21_data is not None and len(s21_data) > 0 else getattr(self.vna.channels, self.current_channel).npts()
                freq_array = np.linspace(start_freq, stop_freq, num_points)
                return freq_array, s21_data
            except (pyvisa.errors.VisaIOError, Exception) as e:
                print(f"Error during VNA sweep on attempt {attempt}: {e}")
                if attempt == max_attempts:
                    raise e
                time.sleep(2)

    def disconnect(self):
        if hasattr(self, "vna"):
            try:
                self.vna.close()
            except Exception as e:
                print(f"Error closing RohdeSchwarzZNB20: {e}")
            finally:
                if hasattr(self, "vna"):
                    del self.vna
        # Also clean up QCodes registry entry for 'VNA'
        try:
            if 'VNA' in Instrument._all_instruments:
                Instrument._all_instruments['VNA'].close()
        except Exception:
            pass
        print("VNA_ZNB20 object connection is closed.")

    def __del__(self):
        print("VNA_ZNB20 object has been deleted")
        try:
            self.disconnect()
        except AttributeError:
            print("self.disconnect() failed.")
            pass

if __name__ == '__main__':
    address = 'TCPIP0::192.168.50.249::inst0::INSTR'
    vna = VNA_ZNB20(address)

    start_freq = 5e9
    stop_freq = 7e9
    points = 1001
    port = "S21"
    power = -20
    IF_bandwidth = 1000

    freq, data = vna.lin_freq_sweep(start_freq, stop_freq, points, port, power, IF_bandwidth)

    plt.plot(freq, 20 * np.log10(np.abs(data)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('S21 Magnitude')
    plt.show()

    vna.disconnect()
