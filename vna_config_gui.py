import os
import re
import socket
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tomlkit

# Configuration files to manage
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_RES_PD = os.path.join(CONFIG_DIR, "resonator_PD.toml")
FILE_WIN_FIND = os.path.join(CONFIG_DIR, "measurement_window_finding.toml")
FILE_MEAS_LF = os.path.join(CONFIG_DIR, "measurement_LF.toml")

def parse_visa_ip(address):
    """Extracts IP address from a VISA string like TCPIP0::192.168.1.123::inst0::INSTR"""
    if not address:
        return ""
    match = re.search(r"TCPIP\d*::([\d\.]+)::", address)
    if match:
        return match.group(1)
    # Fallback to general IP pattern
    match = re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", address)
    if match:
        return match.group(0)
    return address

def build_visa_address(ip):
    """Builds a standard VISA TCP/IP string"""
    ip = ip.strip()
    if ip.startswith("TCPIP"):
        return ip
    return f"TCPIP0::{ip}::inst0::INSTR"

class VnaConfigApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LiteMWSInstr - VNA Configurator")
        self.geometry("1100x680")
        self.configure(bg="#1E1E24")
        self.resizable(True, True)
        self.minsize(1050, 650)

        # Style configurations
        self.setup_styles()
        
        # Build layout
        self.create_widgets()
        
        # Load configs
        self.load_configurations()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Dark Theme Palette
        self.style.configure(".", bg="#1E1E24", fg="#EAEAEA", font=("Segoe UI", 10))
        self.style.configure("TLabel", bg="#1E1E24", foreground="#C5C5C5", font=("Segoe UI", 10, "bold"))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#00A2E8")
        
        self.style.configure("TFrame", bg="#25252D")
        self.style.configure("Card.TFrame", bg="#2C2C35", relief="flat")
        
        # Buttons
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), background="#3E3E4A", foreground="#FFFFFF", borderwidth=0)
        self.style.map("TButton",
            background=[("active", "#4E4E5A"), ("disabled", "#252528")],
            foreground=[("active", "#FFFFFF")]
        )
        
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#007ACC", foreground="#FFFFFF", borderwidth=0)
        self.style.map("Primary.TButton",
            background=[("active", "#0098FF")],
            foreground=[("active", "#FFFFFF")]
        )
        
        self.style.configure("Test.TButton", font=("Segoe UI", 10, "bold"), background="#28A745", foreground="#FFFFFF", borderwidth=0)
        self.style.map("Test.TButton",
            background=[("active", "#34CE57")],
            foreground=[("active", "#FFFFFF")]
        )

        self.style.configure("TCombobox", fieldbackground="#33333D", background="#2C2C35", foreground="#FFFFFF", arrowcolor="#FFFFFF")

    def create_widgets(self):
        # Top banner
        banner = ttk.Frame(self, style="TFrame")
        banner.pack(fill="x", padx=20, pady=(15, 5))
        
        lbl_title = ttk.Label(banner, text="LiteMWSInstr VNA Configuration Tool", style="Header.TLabel")
        lbl_title.pack(side="left")
        
        lbl_status = ttk.Label(banner, text="Ready", font=("Segoe UI", 9, "italic"), foreground="#888888")
        lbl_status.pack(side="right", pady=5)
        self.lbl_status = lbl_status

        # Main Container holding Left and Right panels side-by-side
        main_container = ttk.Frame(self, style="TFrame")
        main_container.pack(fill="both", expand=True, padx=20, pady=5)
        
        # Left Panel (Settings Card)
        left_panel = ttk.Frame(main_container, style="Card.TFrame")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=5)
        
        left_panel.columnconfigure(0, weight=0)
        left_panel.columnconfigure(1, weight=1)
        left_panel.columnconfigure(2, weight=0)

        # Right Panel (Logs & Control Buttons)
        right_panel = ttk.Frame(main_container, style="TFrame")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=5)

        # ---------------- LEFT PANEL: GRID CONFIGURATION ----------------
        # Hardware Section
        section_vna = ttk.Label(left_panel, text="VNA Hardware Settings", style="TLabel")
        section_vna.grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 5))
        
        # IP Address
        ttk.Label(left_panel, text="VNA IP Address:").grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.ent_ip = tk.Entry(left_panel, width=22, font=("Consolas", 11), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_ip.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Quick check buttons
        btn_ping = ttk.Button(left_panel, text="Test Ping", style="Test.TButton", command=self.run_ping_test)
        btn_ping.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        
        # Model selector
        ttk.Label(left_panel, text="VNA Model:").grid(row=2, column=0, sticky="w", padx=15, pady=5)
        self.cmb_model = ttk.Combobox(left_panel, values=["ZNB", "E5080B", "DUMMY"], state="readonly", width=10)
        self.cmb_model.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        btn_idn = ttk.Button(left_panel, text="Query *IDN?", style="Primary.TButton", command=self.run_idn_query)
        btn_idn.grid(row=2, column=2, sticky="w", padx=5, pady=5)

        # Port & Attenuation Settings
        ttk.Label(left_panel, text="Measurement Port:").grid(row=3, column=0, sticky="w", padx=15, pady=5)
        self.ent_port = tk.Entry(left_panel, width=12, font=("Segoe UI", 10), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_port.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(left_panel, text="Attenuation (dB):").grid(row=4, column=0, sticky="w", padx=15, pady=5)
        self.ent_atten = tk.Entry(left_panel, width=12, font=("Segoe UI", 10), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_atten.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Sample Name
        ttk.Label(left_panel, text="Sample Name:").grid(row=5, column=0, sticky="w", padx=15, pady=5)
        self.ent_sample_name = tk.Entry(left_panel, width=22, font=("Segoe UI", 10), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_sample_name.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        # Separator line
        sep = ttk.Separator(left_panel, orient="horizontal")
        sep.grid(row=6, column=0, columnspan=3, sticky="ew", padx=15, pady=10)

        # Output folder Settings
        ttk.Label(left_panel, text="Data Output Paths", style="TLabel").grid(row=7, column=0, columnspan=3, sticky="w", padx=15, pady=(0, 5))
        
        # Path for Resonator PD
        ttk.Label(left_panel, text="Resonator PD Path:").grid(row=8, column=0, sticky="w", padx=15, pady=5)
        self.ent_path_pd = tk.Entry(left_panel, width=35, font=("Consolas", 10), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_path_pd.grid(row=8, column=1, columnspan=2, sticky="we", padx=(5, 15), pady=5)
        
        # Path for Measurement LF
        ttk.Label(left_panel, text="Linear Sweep Path:").grid(row=9, column=0, sticky="w", padx=15, pady=5)
        self.ent_path_lf = tk.Entry(left_panel, width=35, font=("Consolas", 10), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_path_lf.grid(row=9, column=1, columnspan=2, sticky="we", padx=(5, 15), pady=5)

        # Separator line 2
        sep2 = ttk.Separator(left_panel, orient="horizontal")
        sep2.grid(row=10, column=0, columnspan=3, sticky="ew", padx=15, pady=10)

        # Measurement Sweep Parameters Section
        ttk.Label(left_panel, text="Linear Sweep (LF) Sweep Parameters", style="TLabel").grid(row=11, column=0, columnspan=3, sticky="w", padx=15, pady=(0, 5))
        
        sweep_frame = ttk.Frame(left_panel, style="Card.TFrame")
        sweep_frame.grid(row=12, column=0, columnspan=3, sticky="we", padx=15, pady=5)
        
        # Configure columns for sweep_frame
        sweep_frame.columnconfigure(0, weight=1)
        sweep_frame.columnconfigure(1, weight=2)
        sweep_frame.columnconfigure(2, weight=1)
        sweep_frame.columnconfigure(3, weight=2)
        
        # Column 0 & 1: General task params. Column 2 & 3: Freq params
        # Row 0
        ttk.Label(sweep_frame, text="IF Bandwidth (Hz):", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(5, 5), pady=5)
        self.ent_lf_ifbw = tk.Entry(sweep_frame, width=12, font=("Segoe UI", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_lf_ifbw.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(sweep_frame, text="Start Freq (GHz):", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=10, pady=5)
        self.ent_lf_start = tk.Entry(sweep_frame, width=15, font=("Consolas", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_lf_start.grid(row=0, column=3, sticky="we", padx=5, pady=5)
        
        # Row 1
        ttk.Label(sweep_frame, text="Power (dBm):", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", padx=(5, 5), pady=5)
        self.ent_lf_power = tk.Entry(sweep_frame, width=12, font=("Segoe UI", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_lf_power.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(sweep_frame, text="Stop Freq (GHz):", font=("Segoe UI", 9)).grid(row=1, column=2, sticky="w", padx=10, pady=5)
        self.ent_lf_stop = tk.Entry(sweep_frame, width=15, font=("Consolas", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_lf_stop.grid(row=1, column=3, sticky="we", padx=5, pady=5)
        
        # Row 2
        ttk.Label(sweep_frame, text="Repeat Count:", font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", padx=(5, 5), pady=5)
        self.ent_lf_repeat = tk.Entry(sweep_frame, width=12, font=("Segoe UI", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_lf_repeat.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(sweep_frame, text="Sweep Points:", font=("Segoe UI", 9)).grid(row=2, column=2, sticky="w", padx=10, pady=5)
        self.ent_lf_points = tk.Entry(sweep_frame, width=15, font=("Segoe UI", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_lf_points.grid(row=2, column=3, sticky="we", padx=5, pady=5)

        # Row 3
        ttk.Label(sweep_frame, text="Step (MHz):", font=("Segoe UI", 9)).grid(row=3, column=2, sticky="w", padx=10, pady=5)
        self.ent_lf_step = tk.Entry(sweep_frame, width=15, font=("Consolas", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_lf_step.grid(row=3, column=3, sticky="we", padx=5, pady=5)

        # Separator line 3
        sep3 = ttk.Separator(left_panel, orient="horizontal")
        sep3.grid(row=13, column=0, columnspan=3, sticky="ew", padx=15, pady=10)

        # Power-Dependent Sweep (SNR) Section
        ttk.Label(left_panel, text="Power-Dependent Sweep (SNR) Settings", style="TLabel").grid(row=14, column=0, columnspan=3, sticky="w", padx=15, pady=(0, 5))
        
        snr_frame = ttk.Frame(left_panel, style="Card.TFrame")
        snr_frame.grid(row=15, column=0, columnspan=3, sticky="we", padx=15, pady=5)
        
        snr_frame.columnconfigure(0, weight=1)
        snr_frame.columnconfigure(1, weight=2)
        snr_frame.columnconfigure(2, weight=1)
        snr_frame.columnconfigure(3, weight=2)
        
        # Power-Ave Ratio
        ttk.Label(snr_frame, text="Power-Ave Ratio:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(5, 5), pady=5)
        self.ent_snr_ratio = tk.Entry(snr_frame, width=12, font=("Segoe UI", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_snr_ratio.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Max Averages
        ttk.Label(snr_frame, text="Max Averages:", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=10, pady=5)
        self.ent_snr_max = tk.Entry(snr_frame, width=15, font=("Segoe UI", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_snr_max.grid(row=0, column=3, sticky="we", padx=5, pady=5)
        
        # Min Averages
        ttk.Label(snr_frame, text="Min Averages:", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", padx=(5, 5), pady=5)
        self.ent_snr_min = tk.Entry(snr_frame, width=12, font=("Segoe UI", 9), bg="#33333D", fg="#FFFFFF", insertbackground="white", relief="flat")
        self.ent_snr_min.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # ---------------- RIGHT PANEL: LOG & ACTION BUTTONS ----------------
        # Log frame for dark visual aesthetic
        log_frame = ttk.Frame(right_panel, style="Card.TFrame")
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Console logs header & text widget
        ttk.Label(log_frame, text="Console Output & Status Log:", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=15, pady=(15, 5))
        self.txt_log = ScrolledText(log_frame, bg="#18181F", fg="#4AF626", font=("Consolas", 10), relief="flat")
        self.txt_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.txt_log.insert(tk.END, "[System Logs Ready]\n")
        self.txt_log.configure(state="disabled")

        # Bottom buttons panel
        btn_frame = ttk.Frame(right_panel, style="TFrame")
        btn_frame.pack(fill="x", padx=5, pady=(10, 5))
        
        btn_save = ttk.Button(btn_frame, text="Save Settings to Configs", style="Primary.TButton", command=self.save_settings, width=25)
        btn_save.pack(side="right", padx=5)
        
        btn_reload = ttk.Button(btn_frame, text="Reload Configurations", command=self.load_configurations, width=22)
        btn_reload.pack(side="left", padx=5)

        # Bind events for auto-saving
        self.ent_ip.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_ip.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_port.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_port.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_atten.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_atten.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_sample_name.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_sample_name.bind("<Return>", lambda e: self.save_settings(silent=True))

        self.ent_path_pd.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_path_pd.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_path_lf.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_path_lf.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.cmb_model.bind("<<ComboboxSelected>>", lambda e: self.save_settings(silent=True))

        # SNR settings bindings
        self.ent_snr_ratio.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_snr_ratio.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_snr_max.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_snr_max.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_snr_min.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_snr_min.bind("<Return>", lambda e: self.save_settings(silent=True))

        # Sweep parameters bindings
        self.ent_lf_ifbw.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_lf_ifbw.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_lf_power.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_lf_power.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_lf_repeat.bind("<FocusOut>", lambda e: self.save_settings(silent=True))
        self.ent_lf_repeat.bind("<Return>", lambda e: self.save_settings(silent=True))
        
        self.ent_lf_start.bind("<FocusOut>", lambda e: self.on_points_change())
        self.ent_lf_start.bind("<Return>", lambda e: self.on_points_change())
        
        self.ent_lf_stop.bind("<FocusOut>", lambda e: self.on_points_change())
        self.ent_lf_stop.bind("<Return>", lambda e: self.on_points_change())
        
        self.ent_lf_points.bind("<FocusOut>", lambda e: self.on_points_change())
        self.ent_lf_points.bind("<Return>", lambda e: self.on_points_change())

        self.ent_lf_step.bind("<FocusOut>", lambda e: self.on_step_change())
        self.ent_lf_step.bind("<Return>", lambda e: self.on_step_change())

    def log(self, message):
        """Append messages to log area"""
        self.txt_log.configure(state="normal")
        self.txt_log.insert(tk.END, f"{message}\n")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state="disabled")

    def load_configurations(self):
        self.lbl_status.configure(text="Loading configurations...", foreground="#E1A100")
        self.log("\n[Loading settings from TOML config files]")
        
        # 1. Load resonator_PD.toml
        if os.path.exists(FILE_RES_PD):
            try:
                with open(FILE_RES_PD, "r", encoding="utf-8") as f:
                    data = tomlkit.parse(f.read())
                
                hw = data.get("hardware", {})
                address = hw.get("address", "")
                ip = parse_visa_ip(address)
                model = hw.get("model", "")
                port = hw.get("port", "")
                atten = hw.get("attenuation", "")
                
                self.ent_ip.delete(0, tk.END)
                self.ent_ip.insert(0, ip)
                
                if model in ["ZNB", "E5080B", "DUMMY"]:
                    self.cmb_model.set(model)
                else:
                    self.cmb_model.set("DUMMY")
                
                self.ent_port.delete(0, tk.END)
                self.ent_port.insert(0, str(port))
                
                self.ent_atten.delete(0, tk.END)
                self.ent_atten.insert(0, str(atten))
                
                out = data.get("output", {})
                data_path = out.get("data_path", "")
                self.ent_path_pd.delete(0, tk.END)
                self.ent_path_pd.insert(0, str(data_path))
                
                # Load sample name
                sample = data.get("sample", {})
                sample_name = sample.get("name", "")
                self.ent_sample_name.delete(0, tk.END)
                self.ent_sample_name.insert(0, str(sample_name))
                
                # Load SNR parameters from first resonator entry
                resonators = data.get("resonator", [])
                if resonators and len(resonators) > 0:
                    r0 = resonators[0]
                    snr = r0.get("snr", {})
                    power_ave_ratio = snr.get("power_ave_ratio", "")
                    max_repeat = snr.get("max_repeat", "")
                    min_repeat = snr.get("min_repeat", "")
                    
                    self.ent_snr_ratio.delete(0, tk.END)
                    self.ent_snr_ratio.insert(0, str(power_ave_ratio))
                    
                    self.ent_snr_max.delete(0, tk.END)
                    self.ent_snr_max.insert(0, str(max_repeat))
                    
                    self.ent_snr_min.delete(0, tk.END)
                    self.ent_snr_min.insert(0, str(min_repeat))
                
                self.log(f"-> Loaded {os.path.basename(FILE_RES_PD)} (IP={ip}, Model={model})")
            except Exception as e:
                self.log(f"Error loading {FILE_RES_PD}: {e}")
        else:
            self.log(f"Warning: {FILE_RES_PD} not found.")

        # 2. Load measurement_LF.toml
        if os.path.exists(FILE_MEAS_LF):
            try:
                with open(FILE_MEAS_LF, "r", encoding="utf-8") as f:
                    data = tomlkit.parse(f.read())
                
                meas = data.get("measurement", [])
                if meas and len(meas) > 0:
                    m = meas[0]
                    output_path = m.get("output", "")
                    self.ent_path_lf.delete(0, tk.END)
                    self.ent_path_lf.insert(0, str(output_path))
                    
                    ifbw = m.get("IF_bandwidth", "")
                    self.ent_lf_ifbw.delete(0, tk.END)
                    self.ent_lf_ifbw.insert(0, str(ifbw))
                    
                    power = m.get("power", "")
                    self.ent_lf_power.delete(0, tk.END)
                    self.ent_lf_power.insert(0, str(power))
                    
                    repeat = m.get("repeat", "")
                    self.ent_lf_repeat.delete(0, tk.END)
                    self.ent_lf_repeat.insert(0, str(repeat))
                    
                    freq = m.get("frequency", {})
                    start = freq.get("start", "")
                    if start != "":
                        try:
                            start = f"{float(start) / 1e9:.6f}".rstrip('0').rstrip('.')
                        except ValueError:
                            pass
                    self.ent_lf_start.delete(0, tk.END)
                    self.ent_lf_start.insert(0, str(start))
                    
                    stop = freq.get("stop", "")
                    if stop != "":
                        try:
                            stop = f"{float(stop) / 1e9:.6f}".rstrip('0').rstrip('.')
                        except ValueError:
                            pass
                    self.ent_lf_stop.delete(0, tk.END)
                    self.ent_lf_stop.insert(0, str(stop))
                    
                    points = freq.get("points", "")
                    self.ent_lf_points.delete(0, tk.END)
                    self.ent_lf_points.insert(0, str(points))
                    
                    # Update step size based on loaded start, stop, and points
                    self.update_step_from_points()
                
                self.log(f"-> Loaded {os.path.basename(FILE_MEAS_LF)}")
            except Exception as e:
                self.log(f"Error loading {FILE_MEAS_LF}: {e}")
        else:
            self.log(f"Warning: {FILE_MEAS_LF} not found.")

        self.lbl_status.configure(text="Config Loaded", foreground="#28A745")

    def run_ping_test(self):
        ip = self.ent_ip.get().strip()
        if not ip:
            messagebox.showwarning("Warning", "Please enter VNA IP address first!")
            return
        
        self.lbl_status.configure(text="Pinging VNA...", foreground="#E1A100")
        self.log(f"\n[Ping Test] Sending ICMP echo to {ip}...")
        
        def ping_thread():
            try:
                # Option -n 2 for Windows (ping 2 times), -c 2 for Mac/Linux
                ping_cmd = ["ping", "-n", "2", ip] if os.name == "nt" else ["ping", "-c", "2", ip]
                res = subprocess.run(ping_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                self.log(res.stdout)
                if res.returncode == 0:
                    self.log("Ping successful! Network connection is active.")
                    self.lbl_status.configure(text="Ping OK", foreground="#28A745")
                else:
                    self.log("Ping failed. Please check network settings, IP address, or cable connection.")
                    self.lbl_status.configure(text="Ping Failed", foreground="#DC3545")
            except Exception as e:
                self.log(f"Error executing ping: {e}")
                self.lbl_status.configure(text="Error", foreground="#DC3545")

        threading.Thread(target=ping_thread, daemon=True).start()

    def run_idn_query(self):
        ip = self.ent_ip.get().strip()
        if not ip:
            messagebox.showwarning("Warning", "Please enter VNA IP address first!")
            return
        
        self.lbl_status.configure(text="Querying *IDN?...", foreground="#E1A100")
        self.log(f"\n[Connection Query] Querying *IDN? from {ip}:5025 (TCP raw socket)...")
        
        def idn_thread():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4.0)
                s.connect((ip, 5025))
                s.sendall(b"*IDN?\n")
                response = s.recv(1024).decode().strip()
                s.close()
                
                self.log(f"Received reply: {response}")
                self.lbl_status.configure(text="Query Success", foreground="#28A745")
                
                # Auto detect model
                if "ZNB" in response:
                    self.cmb_model.set("ZNB")
                    self.log("Detected Rohde & Schwarz ZNB. Suggest model: 'ZNB' (Auto selected).")
                elif "E5080" in response:
                    self.cmb_model.set("E5080B")
                    self.log("Detected Keysight E5080B. Suggest model: 'E5080B' (Auto selected).")
                else:
                    self.log("Unknown instrument type. Suggest manually selecting VNA model.")
                
                # Auto-save changes immediately on GUI thread
                self.after(0, lambda: self.save_settings(silent=True))
            except socket.timeout:
                self.log("Connection Timeout. VNA did not respond within 4 seconds over port 5025.")
                self.lbl_status.configure(text="Timeout", foreground="#DC3545")
            except Exception as e:
                self.log(f"Connection failed: {e}. Make sure port 5025 is enabled on VNA.")
                self.lbl_status.configure(text="Connect Error", foreground="#DC3545")

        threading.Thread(target=idn_thread, daemon=True).start()

    def save_settings(self, silent=False):
        ip = self.ent_ip.get().strip()
        model = self.cmb_model.get()
        port = self.ent_port.get().strip()
        atten_val = self.ent_atten.get().strip()
        path_pd = self.ent_path_pd.get().strip()
        path_lf = self.ent_path_lf.get().strip()
        sample_name = self.ent_sample_name.get().strip()

        lf_ifbw_val = self.ent_lf_ifbw.get().strip()
        lf_power_val = self.ent_lf_power.get().strip()
        lf_repeat_val = self.ent_lf_repeat.get().strip()
        lf_start_val = self.ent_lf_start.get().strip()
        lf_stop_val = self.ent_lf_stop.get().strip()
        lf_points_val = self.ent_lf_points.get().strip()

        snr_ratio_val = self.ent_snr_ratio.get().strip()
        snr_max_val = self.ent_snr_max.get().strip()
        snr_min_val = self.ent_snr_min.get().strip()

        if not ip:
            if not silent:
                messagebox.showwarning("Warning", "IP Address cannot be empty!")
            else:
                self.log("Auto-save skipped: IP Address is empty.")
            return
        
        try:
            atten = int(atten_val) if atten_val else 0
        except ValueError:
            if not silent:
                messagebox.showerror("Error", "Attenuation must be an integer!")
            else:
                self.log("Auto-save skipped: Attenuation must be an integer.")
            return

        try:
            lf_ifbw = int(lf_ifbw_val) if lf_ifbw_val else 1000
            lf_power = float(lf_power_val) if lf_power_val else -20.0
            lf_repeat = int(lf_repeat_val) if lf_repeat_val else 1
            lf_start = float(lf_start_val) * 1e9 if lf_start_val else 4e9
            lf_stop = float(lf_stop_val) * 1e9 if lf_stop_val else 8e9
            lf_points = int(lf_points_val) if lf_points_val else 501
        except ValueError as e:
            if not silent:
                messagebox.showerror("Error", f"Invalid measurement sweep parameters: {e}")
            else:
                self.log(f"Auto-save skipped: Invalid sweep parameter value ({e}).")
            return

        try:
            snr_ratio = int(snr_ratio_val) if snr_ratio_val else -140
            snr_max = int(snr_max_val) if snr_max_val else 5
            snr_min = int(snr_min_val) if snr_min_val else 1
        except ValueError as e:
            if not silent:
                messagebox.showerror("Error", f"Invalid SNR settings: {e}")
            else:
                self.log(f"Auto-save skipped: Invalid SNR parameter value ({e}).")
            return

        address = build_visa_address(ip)
        if silent:
            self.log("[Auto-saving config changes to files]")
        else:
            self.log("\n[Saving config changes to files]")

        # 1. Update resonator_PD.toml
        if os.path.exists(FILE_RES_PD):
            try:
                with open(FILE_RES_PD, "r", encoding="utf-8") as f:
                    content = tomlkit.parse(f.read())
                
                if "hardware" not in content:
                    content["hardware"] = tomlkit.table()
                content["hardware"]["model"] = model
                content["hardware"]["address"] = address
                content["hardware"]["port"] = port
                content["hardware"]["attenuation"] = atten
                
                if "sample" not in content:
                    content["sample"] = tomlkit.table()
                content["sample"]["name"] = sample_name

                if "output" not in content:
                    content["output"] = tomlkit.table()
                content["output"]["data_path"] = path_pd
                
                # Update all resonators globally with the SNR parameters
                if "resonator" in content:
                    for res in content["resonator"]:
                        if "snr" not in res:
                            res["snr"] = tomlkit.table()
                        res["snr"]["power_ave_ratio"] = snr_ratio
                        res["snr"]["max_repeat"] = snr_max
                        res["snr"]["min_repeat"] = snr_min
                
                with open(FILE_RES_PD, "w", encoding="utf-8") as f:
                    f.write(tomlkit.dumps(content))
                self.log(f"Successfully updated {os.path.basename(FILE_RES_PD)}")
            except Exception as e:
                self.log(f"Failed to update {FILE_RES_PD}: {e}")

        # 2. Update measurement_window_finding.toml
        if os.path.exists(FILE_WIN_FIND):
            try:
                with open(FILE_WIN_FIND, "r", encoding="utf-8") as f:
                    content = tomlkit.parse(f.read())
                
                if "hardware" not in content:
                    content["hardware"] = tomlkit.table()
                content["hardware"]["model"] = model
                content["hardware"]["address"] = address
                content["hardware"]["port"] = port
                content["hardware"]["attenuation"] = atten
                
                if "sample" not in content:
                    content["sample"] = tomlkit.table()
                content["sample"]["name"] = sample_name
                
                with open(FILE_WIN_FIND, "w", encoding="utf-8") as f:
                    f.write(tomlkit.dumps(content))
                self.log(f"Successfully updated {os.path.basename(FILE_WIN_FIND)}")
            except Exception as e:
                self.log(f"Failed to update {FILE_WIN_FIND}: {e}")

        # 3. Update measurement_LF.toml
        if os.path.exists(FILE_MEAS_LF):
            try:
                with open(FILE_MEAS_LF, "r", encoding="utf-8") as f:
                    content = tomlkit.parse(f.read())
                
                if "hardware" not in content:
                    content["hardware"] = tomlkit.table()
                content["hardware"]["model"] = model
                content["hardware"]["address"] = address
                content["hardware"]["port"] = port
                content["hardware"]["attenuation"] = atten
                
                if "sample" not in content:
                    content["sample"] = tomlkit.table()
                content["sample"]["name"] = sample_name
                
                if "measurement" in content and len(content["measurement"]) > 0:
                    for i in range(len(content["measurement"])):
                        content["measurement"][i]["output"] = path_lf
                        content["measurement"][i]["IF_bandwidth"] = lf_ifbw
                        content["measurement"][i]["power"] = lf_power
                        content["measurement"][i]["repeat"] = lf_repeat
                        if "frequency" not in content["measurement"][i]:
                            content["measurement"][i]["frequency"] = tomlkit.table()
                        content["measurement"][i]["frequency"]["start"] = lf_start
                        content["measurement"][i]["frequency"]["stop"] = lf_stop
                        content["measurement"][i]["frequency"]["points"] = lf_points
                
                with open(FILE_MEAS_LF, "w", encoding="utf-8") as f:
                    f.write(tomlkit.dumps(content))
                self.log(f"Successfully updated {os.path.basename(FILE_MEAS_LF)}")
            except Exception as e:
                self.log(f"Failed to update {FILE_MEAS_LF}: {e}")

        self.lbl_status.configure(text="Settings Saved", foreground="#28A745")
        if not silent:
            messagebox.showinfo("Success", "All configuration files updated successfully!")
        else:
            self.log("-> Auto-saved settings successfully.")

    def update_step_from_points(self):
        try:
            start = float(self.ent_lf_start.get().strip())
            stop = float(self.ent_lf_stop.get().strip())
            points = int(self.ent_lf_points.get().strip())
            if points > 1:
                # start and stop are in GHz, calculate step in MHz
                step_mhz = ((stop - start) * 1000.0) / (points - 1)
                self.ent_lf_step.delete(0, tk.END)
                self.ent_lf_step.insert(0, f"{step_mhz:.4f}".rstrip('0').rstrip('.'))
        except ValueError:
            pass

    def update_points_from_step(self):
        try:
            start = float(self.ent_lf_start.get().strip())
            stop = float(self.ent_lf_stop.get().strip())
            step_mhz = float(self.ent_lf_step.get().strip())
            if step_mhz > 0:
                # start and stop are in GHz, calculate points
                points = int(round(((stop - start) * 1000.0) / step_mhz)) + 1
                if points < 2:
                    points = 2
                self.ent_lf_points.delete(0, tk.END)
                self.ent_lf_points.insert(0, str(points))
        except ValueError:
            pass

    def on_points_change(self):
        self.update_step_from_points()
        self.save_settings(silent=True)

    def on_step_change(self):
        self.update_points_from_step()
        self.save_settings(silent=True)

if __name__ == "__main__":
    app = VnaConfigApp()
    app.mainloop()
