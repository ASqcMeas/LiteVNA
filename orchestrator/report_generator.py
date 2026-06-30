import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

class ReportGenerator:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def save_sweep_netcdf(self, file_path, freq_array, s_params, attrs, vna_port):
        """
        Saves raw sweep S-parameters to a NetCDF file.
        """
        try:
            var_key = str(vna_port).lower()
            output_data = {
                var_key: (["s_params", "frequency"], np.array([s_params.real, s_params.imag]))
            }
            dataset = xr.Dataset(
                output_data,
                coords={"s_params": np.array(["real", "imag"]), "frequency": freq_array}
            )
            for k, v in attrs.items():
                dataset.attrs[k] = v
                
            # Ensure folder exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            dataset.to_netcdf(file_path)
            print(f"  [Save Data] Saved sweep NetCDF: {file_path}")
        except Exception as save_err:
            print(f"  Warning: Failed to save sweep NetCDF to {file_path}: {save_err}")

    def generate_audit_csv(self, base_data_dir, start_freq, stop_freq, search_report, configs):
        """
        Generates standard CSV and pretty Markdown audit trail reports.
        """
        try:
            os.makedirs(base_data_dir, exist_ok=True)
            csv_path = os.path.join(base_data_dir, "blind_search_report.csv")
            md_path = os.path.join(base_data_dir, "blind_search_report.md")
            
            print(f"\nGenerating blind search audit reports at:\n  CSV: {csv_path}\n  MD:  {md_path}")
            
            # 1. Gather DataFrame and Columns
            cols = [
                "Type", "Coarse_Frequency_GHz", "Refined_Frequency_GHz", "Power_dBm",
                "IF_Bandwidth_Hz", "Points", "Est_Noise_Std_dB", "Prominence_Floor_dB",
                "Start_Frequency_GHz", "Stop_Frequency_GHz",
                "FWHM_MHz", "FWHM_fit_MHz", "Depth_dB", "Qi_fit", "ChiSq_fit", "Confidence_Score", 
                "Status", "Reason", "Next_Sweep_Start_GHz", "Next_Sweep_Stop_GHz"
            ]
            
            if search_report:
                df = pd.DataFrame(search_report)
                for col in cols:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[cols]
            else:
                df = pd.DataFrame(columns=cols)
                
            # 2. Save pure RFC-compliant CSV (no comments)
            df.to_csv(csv_path, index=False, na_rep="NaN")
            print(f"Successfully saved standard CSV report to: {csv_path}")
            
            # 3. Save pretty Markdown report with configurations metadata and separated tables
            sample_name = self.config_manager.res_pd_config.get("sample", {}).get("name", "resonator")
            
            # Calculate summary statistics
            total_candidates = len(df)
            passed_df = df[df["Status"] == "Passed"].copy()
            discarded_df = df[df["Status"] == "Discarded"].copy()
            
            score_filtered_df = discarded_df[discarded_df["Type"] == "Score Filtered"].copy()
            other_discarded_df = discarded_df[discarded_df["Type"] != "Score Filtered"].copy()
            
            # Sorting
            if not passed_df.empty:
                sort_col = "Refined_Frequency_GHz" if "Refined_Frequency_GHz" in passed_df.columns and not passed_df["Refined_Frequency_GHz"].isna().all() else "Coarse_Frequency_GHz"
                passed_df.sort_values(by=sort_col, inplace=True)
            if not score_filtered_df.empty:
                score_filtered_df.sort_values(by="Confidence_Score", ascending=False, inplace=True)
            if not other_discarded_df.empty:
                sort_col = "Coarse_Frequency_GHz" if "Coarse_Frequency_GHz" in other_discarded_df.columns else "Type"
                other_discarded_df.sort_values(by=sort_col, inplace=True)

            reason_counts = discarded_df["Type"].value_counts().to_dict() if not discarded_df.empty else {}

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write("# VNA 共振腔盲搜尋與驗證審計報告 (Blind Search & Verification Report)\n\n")
                f.write(f"* **量測時間 (Generated At)**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"* **樣品名稱 (Sample Name)**: `{sample_name}`\n")
                f.write(f"* **搜尋頻帶 (Search Range)**: `{start_freq/1e9:.3f}` 至 `{stop_freq/1e9:.3f}` GHz\n\n")
                
                f.write("---\n\n")
                f.write("## 1. 量測與篩選門檻設定 (Configurations & Thresholds)\n\n")
                f.write("| 設定項目 (Parameter) | 設定值 (Value) | 說明 (Description) |\n")
                f.write("| :--- | :--- | :--- |\n")
                f.write(f"| **評分計算模式 (Scoring Method)** | `{configs.get('scoring_method', 'geometric')}` | 選擇總分計算算法 (`geometric` 幾何平均 / `arithmetic` 算術平均) |\n")
                f.write(f"| **幾何分數門檻 (Min Geometric Score)** | `{configs.get('min_geometric_score', 'N/A')}` | 幾何平均滿分 100 分之通過下限門檻 |\n")
                f.write(f"| **算術分數門檻 (Min Arithmetic Score)** | `{configs.get('min_arithmetic_score', 'N/A')}` | 算術平均滿分 100 分之通過下限門檻 |\n")
                f.write(f"| **圓擬合殘差上限 (Max ChiSq Fit)** | `{configs.get('max_chisq_fit', 'N/A')}` | 物理圓擬合殘差 $\\chi^2$ 品質硬性上限 |\n")
                f.write(f"| **FWHM 最大不一致倍數 (Max FWHM Ratio)** | `{configs.get('max_fwhm_ratio', 'N/A')}` | 擬合 FWHM 與光譜 FWHM 最大允許比例倍數 |\n")
                f.write(f"| **FWHM 硬性下限 (Min FWHM kHz)** | `{configs.get('min_fwhm_khz', 'N/A')}` | 細掃驗證 FWHM 硬性下限 (kHz) |\n")
                f.write(f"| **FWHM 硬性上限 (Max FWHM MHz)** | `{configs.get('max_fwhm_mhz', 'N/A')}` | 細掃驗證 FWHM 硬性上限 (MHz) |\n")
                f.write(f"| **評分滿分中心寬度 (Target FWHM)** | `{configs.get('target_fwhm', 0.0)/1e3:.1f} kHz` | 評分系統滿分基準 FWHM |\n")
                f.write(f"| **FWHM 對數標準差 (Sigma Decade)** | `{configs.get('sigma_dec', 0.0):.3f}` | 寬度懲罰對數容忍標準差 |\n")
                f.write(f"| **尋峰降噪倍數 (Noise Sigma Mult)** | `{configs.get('ns_mult', 0.0):.1f}` | 粗掃動態 Prominence 閥值標準差倍數 |\n")
                f.write(f"| **視窗擴展倍數 (Window Multiplier)** | `{configs.get('window_multiplier', 0.0):.1f}` | 最終量測視窗相對於 FWHM 之倍數 |\n")
                f.write(f"| **粗掃去重間距 (Coarse Spacing)** | `{configs.get('coarse_spacing', 0.0)/1e6:.3f} MHz` | 粗掃階段候選點最小允許間距 |\n")
                f.write(f"| **精細去重間距 (Precise Spacing)** | `{configs.get('precise_spacing', 0.0)/1e6:.3f} MHz` | 驗證階段共振腔最小允許間距 |\n\n")
                
                f.write("---\n\n")
                f.write("## 2. 統計審計總覽 (Executive Summary & Statistics)\n\n")
                f.write(f"* 🔍 **粗掃探測候選點總數 (Total Candidates Found)**：**{total_candidates}** 個\n")
                f.write(f"* 🟢 **最終驗證成功通過 (Successfully Verified Resonators)**：**{len(passed_df)}** 個\n")
                f.write(f"* 🔴 **過濾與淘汰總數 (Total Discarded Candidates)**：**{len(discarded_df)}** 個\n\n")
                
                if reason_counts:
                    f.write("### 📉 淘汰原因詳細統計拆解 (Discarded Breakdown)\n\n")
                    f.write("| 淘汰分類 (Discard Reason Type) | 數量 (Count) | 比例 (Percentage) |\n")
                    f.write("| :--- | :---: | :---: |\n")
                    for r_type, r_count in reason_counts.items():
                        pct = (r_count / total_candidates) * 100 if total_candidates > 0 else 0
                        f.write(f"| `{r_type}` | **{r_count}** | {pct:.1f}% |\n")
                    f.write("\n")

                f.write("---\n\n")
                f.write(f"## 3. 🟢 通過驗證的共振腔 (Passed Resonators - 共 {len(passed_df)} 個)\n\n")
                if not passed_df.empty:
                    f.write("| 標籤 (Label) | 精確驗證頻率 (GHz) | 粗掃頻率 (GHz) | 功率 (dBm) | 光譜 FWHM (MHz) | 擬合 FWHM (MHz) | 深度 (dB) | $Q_i$ 擬合值 | $\\chi^2$ 殘差 | 信心度得分 |\n")
                    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                    for _, row in passed_df.iterrows():
                        label = f"C{int(row['Refined_Frequency_GHz']*1e4)}" if not pd.isna(row['Refined_Frequency_GHz']) else "N/A"
                        ref_f = f"{row['Refined_Frequency_GHz']:.6f}" if not pd.isna(row['Refined_Frequency_GHz']) else "N/A"
                        crs_f = f"{row['Coarse_Frequency_GHz']:.6f}" if not pd.isna(row['Coarse_Frequency_GHz']) else "N/A"
                        pwr = f"{row['Power_dBm']:.1f}" if not pd.isna(row['Power_dBm']) else "N/A"
                        fwhm = f"{row['FWHM_MHz']:.3f}" if not pd.isna(row['FWHM_MHz']) else "N/A"
                        fwhm_fit = f"{row['FWHM_fit_MHz']:.3f}" if not pd.isna(row['FWHM_fit_MHz']) else "N/A"
                        dpth = f"{row['Depth_dB']:.2f}" if not pd.isna(row['Depth_dB']) else "N/A"
                        qi = f"{row['Qi_fit']:.1f}" if not pd.isna(row['Qi_fit']) else "N/A"
                        chisq = f"{row['ChiSq_fit']:.5f}" if not pd.isna(row['ChiSq_fit']) else "N/A"
                        score = f"**{row['Confidence_Score']:.1f}**" if not pd.isna(row['Confidence_Score']) else "N/A"
                        f.write(f"| `{label}` | {ref_f} | {crs_f} | {pwr} | {fwhm} | {fwhm_fit} | {dpth} | {qi} | {chisq} | {score} |\n")
                    f.write("\n")
                else:
                    f.write("*無任何共振腔通過驗證。*\n\n")

                f.write("---\n\n")
                f.write(f"## 4. 🟡 因分數未達門檻而被淘汰的訊號 (Score Filtered Candidates - 共 {len(score_filtered_df)} 個)\n\n")
                if not score_filtered_df.empty:
                    f.write("| 粗掃頻率 (GHz) | 驗證頻率 (GHz) | 功率 (dBm) | 光譜 FWHM (MHz) | 擬合 FWHM (MHz) | 深度 (dB) | $Q_i$ 擬合值 | $\\chi^2$ 殘差 | 得分 (Score) | 淘汰詳細原因 (Reason) |\n")
                    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                    for _, row in score_filtered_df.iterrows():
                        ref_f = f"{row['Refined_Frequency_GHz']:.6f}" if not pd.isna(row['Refined_Frequency_GHz']) else "N/A"
                        crs_f = f"{row['Coarse_Frequency_GHz']:.6f}" if not pd.isna(row['Coarse_Frequency_GHz']) else "N/A"
                        pwr = f"{row['Power_dBm']:.1f}" if not pd.isna(row['Power_dBm']) else "N/A"
                        fwhm = f"{row['FWHM_MHz']:.3f}" if not pd.isna(row['FWHM_MHz']) else "N/A"
                        fwhm_fit = f"{row['FWHM_fit_MHz']:.3f}" if not pd.isna(row['FWHM_fit_MHz']) else "N/A"
                        dpth = f"{row['Depth_dB']:.2f}" if not pd.isna(row['Depth_dB']) else "N/A"
                        qi = f"{row['Qi_fit']:.1f}" if not pd.isna(row['Qi_fit']) else "N/A"
                        chisq = f"{row['ChiSq_fit']:.5f}" if not pd.isna(row['ChiSq_fit']) else "N/A"
                        score = f"{row['Confidence_Score']:.1f}" if not pd.isna(row['Confidence_Score']) else "N/A"
                        reason = str(row['Reason'])
                        f.write(f"| {crs_f} | {ref_f} | {pwr} | {fwhm} | {fwhm_fit} | {dpth} | {qi} | {chisq} | {score} | {reason} |\n")
                    f.write("\n")
                else:
                    f.write("*無任何訊號因分數未達標被淘汰。*\n\n")

                f.write("---\n\n")
                f.write(f"## 5. 🔴 其他原因被過濾或失敗的訊號 (Other Discarded Candidates - 共 {len(other_discarded_df)} 個)\n\n")
                if not other_discarded_df.empty:
                    f.write("| 淘汰類型 (Type) | 粗掃頻率 (GHz) | 驗證頻率 (GHz) | 功率 (dBm) | 光譜 FWHM (MHz) | 擬合 FWHM (MHz) | 深度 (dB) | 淘汰詳細原因 (Reason) |\n")
                    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                    for _, row in other_discarded_df.iterrows():
                        t_type = f"`{row['Type']}`"
                        ref_f = f"{row['Refined_Frequency_GHz']:.6f}" if not pd.isna(row['Refined_Frequency_GHz']) else "N/A"
                        crs_f = f"{row['Coarse_Frequency_GHz']:.6f}" if not pd.isna(row['Coarse_Frequency_GHz']) else "N/A"
                        pwr = f"{row['Power_dBm']:.1f}" if not pd.isna(row['Power_dBm']) else "N/A"
                        fwhm = f"{row['FWHM_MHz']:.3f}" if not pd.isna(row['FWHM_MHz']) else "N/A"
                        fwhm_fit = f"{row['FWHM_fit_MHz']:.3f}" if not pd.isna(row['FWHM_fit_MHz']) else "N/A"
                        dpth = f"{row['Depth_dB']:.2f}" if not pd.isna(row['Depth_dB']) else "N/A"
                        reason = str(row['Reason'])
                        f.write(f"| {t_type} | {crs_f} | {ref_f} | {pwr} | {fwhm} | {fwhm_fit} | {dpth} | {reason} |\n")
                    f.write("\n")
                else:
                    f.write("*無其他原因淘汰之訊號。*\n\n")

                f.write("---\n\n")
                f.write("## 6. 完整原始歷程紀錄表 (Full Audit Trail Raw Data)\n\n")
                headers = list(df.columns)
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for _, row in df.iterrows():
                    row_vals = []
                    for val in row:
                        if pd.isna(val):
                            row_vals.append("NaN")
                        elif isinstance(val, float):
                            row_vals.append(f"{val:.6f}")
                        else:
                            row_vals.append(str(val))
                    f.write("| " + " | ".join(row_vals) + " |\n")
                    
            print(f"Successfully saved Markdown report to: {md_path}")
            
        except Exception as err:
            print(f"Warning: Failed to generate reports: {err}")

    def generate_plots(self, base_data_dir, all_verification_candidates, refined_resonators, cached_sweeps, sweep_passes, vna_port):
        """
        Generates visualization plots for all candidates (both passed and discarded).
        Plots are organized into plots/passed and plots/discarded subfolders.
        """
        if (len(all_verification_candidates) > 0 or len(refined_resonators) > 0) and len(cached_sweeps) > 0:
            try:
                plots_dir = os.path.join(base_data_dir, "plots")
                passed_dir = os.path.join(plots_dir, "passed")
                discarded_dir = os.path.join(plots_dir, "discarded")
                os.makedirs(passed_dir, exist_ok=True)
                os.makedirs(discarded_dir, exist_ok=True)
                print(f"Created base output directories for plots: {plots_dir} (passed & discarded)")
 
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
                coarse_path = os.path.join(plots_dir, "blind_search_coarse.png")
                plt.savefig(coarse_path, dpi=150)
                plt.close()
                print(f"Saved coarse sweep plot to: {coarse_path}")

                # 1b. Save Raw Coarse Sweep Plot (blind_search_coarse_raw.png - no vertical lines)
                plt.figure(figsize=(10, 5))
                for p_idx, (freq_c, spar_c) in enumerate(cached_sweeps):
                    mag_c = 20 * np.log10(np.maximum(np.abs(spar_c), 1e-18))
                    p_name = sweep_passes[p_idx]["name"]
                    color = colors[p_idx % len(colors)]
                    plt.plot(freq_c / 1e9, mag_c, color=color, alpha=0.7 - 0.15 * min(p_idx, 2), label=p_name)
                    
                plt.xlabel("Frequency (GHz)")
                plt.ylabel(f"{vna_port} Magnitude (dB)")
                plt.title("Blind Resonator Search - Raw Coarse Sweeps")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                coarse_raw_path = os.path.join(plots_dir, "blind_search_coarse_raw.png")
                plt.savefig(coarse_raw_path, dpi=150)
                plt.close()
                print(f"Saved raw coarse sweep plot to: {coarse_raw_path}")

                # 1c. Save individual plots for each coarse sweep pass (no vertical lines)
                for p_idx, (freq_c, spar_c) in enumerate(cached_sweeps):
                    plt.figure(figsize=(10, 5))
                    mag_c = 20 * np.log10(np.maximum(np.abs(spar_c), 1e-18))
                    p_name = sweep_passes[p_idx]["name"]
                    color = colors[p_idx % len(colors)]
                    plt.plot(freq_c / 1e9, mag_c, color=color, alpha=0.7, label=p_name)
                    
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel(f"{vna_port} Magnitude (dB)")
                    plt.title(f"Blind Resonator Search - Coarse Sweep ({p_name})")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    pass_path = os.path.join(plots_dir, f"blind_search_coarse_pass{p_idx+1}.png")
                    plt.savefig(pass_path, dpi=150)
                    plt.close()
                    print(f"Saved coarse sweep pass {p_idx+1} plot to: {pass_path}")
                
                # 2. Save Combined Fine Sweeps Plot (blind_search_fine_all.png)
                plt.figure(figsize=(10, 5))
                for r in refined_resonators:
                    if "freq_v" in r:
                        mag_v = 20 * np.log10(np.maximum(np.abs(r["s21_v"]), 1e-18))
                        offset = r["freq_v"] - r["design_freq"]
                        x_start_v = max(r["start"], r["freq_v"].min())
                        x_stop_v = max(r["stop"], r["freq_v"].max())
                        mask = (r["freq_v"] >= x_start_v) & (r["freq_v"] <= x_stop_v)
                        if np.any(mask):
                            plt.plot(offset[mask] / 1e6, mag_v[mask], 
                                     label=f"{r['label']} ({r['design_freq']/1e9:.5f} GHz)")
                        
                plt.xlabel("Frequency Offset from Center (MHz)")
                plt.ylabel(f"{vna_port} Magnitude (dB)")
                plt.title("Verified Resonators - Combined Fine Sweeps (Adaptive Zoom)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fine_all_path = os.path.join(plots_dir, "blind_search_fine_all.png")
                plt.savefig(fine_all_path, dpi=150)
                plt.close()
                print(f"Saved combined fine sweeps plot to: {fine_all_path}")
                
                # 3. Save individual fine sweeps and IQ circle plots for all candidates (separated into passed and discarded)
                for r in all_verification_candidates:
                    if "freq_v" in r:
                        status = r.get("status", "Discarded")
                        target_dir = passed_dir if status == "Passed" else discarded_dir
                        
                        plt.figure(figsize=(7, 4.5))
                        mag_v = 20 * np.log10(np.maximum(np.abs(r["s21_v"]), 1e-18))
                        
                        f_center_ghz = r["design_freq"] / 1e9
                        plt.plot(r["freq_v"] / 1e9, mag_v, '.', color="#3498db", alpha=0.6, label="rawdata")
                        
                        # Overlay fitted S21 curve if available
                        if r.get("s21_sim") is not None:
                            mag_sim = 20 * np.log10(np.maximum(np.abs(r["s21_sim"]), 1e-18))
                            plt.plot(r["freq_v"] / 1e9, mag_sim, '-', color="#e74c3c", linewidth=2.0, label="fit")

                        plt.axvline(f_center_ghz, color="#e74c3c", linestyle="--", 
                                    label=f"Center ({f_center_ghz:.5f} GHz)")
                        
                        # Calculate FWHM horizontal helper line
                        y_base = np.max(mag_v)
                        y_dip = np.min(mag_v)
                        y_half = (y_base + y_dip) / 2.0
                        
                        fwhm_hz = r["fwhm"]
                        fwhm_mhz = fwhm_hz / 1e6
                        x_min = f_center_ghz - (fwhm_hz / 2.0) / 1e9
                        x_max = f_center_ghz + (fwhm_hz / 2.0) / 1e9
                        
                        plt.hlines(y=y_half, xmin=x_min, xmax=x_max, colors="#2ecc71", linewidth=2.0,
                                   label=f"FWHM: {fwhm_mhz:.3f} MHz")
                        plt.plot([x_min, x_max], [y_half, y_half], 'o', color="#2ecc71", markersize=5)
                        
                        x_start = max(r["start"], r["freq_v"].min())
                        x_stop = min(r["stop"], r["freq_v"].max())
                        plt.xlim(x_start / 1e9, x_stop / 1e9)
                        
                        plt.xlabel("Frequency (GHz)")
                        plt.ylabel(f"{vna_port} Magnitude (dB)")
                        plt.title(f"Verification Fine Sweep - {r['label']} [{status}] (Adaptive Zoom)")
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        indiv_path = os.path.join(target_dir, f"blind_search_fine_{r['label']}.png")
                        plt.savefig(indiv_path, dpi=120)
                        plt.close()
                        print(f"Saved individual fine sweep plot ({status}) to: {indiv_path}")
                        
                        # Save IQ Circle Plot for Verification
                        if r.get("z_data_raw") is not None and r.get("s21_sim") is not None:
                            plt.figure(figsize=(6, 5))
                            raw_z = r["z_data_raw"]
                            sim_z = r["s21_sim"]
                            plt.plot(raw_z.real, raw_z.imag, '.', label="rawdata", color="#3498db", alpha=0.6)
                            plt.plot(sim_z.real, sim_z.imag, '-', label="fit", color="#e74c3c", linewidth=2.0)
                            plt.xlabel("Re(S)")
                            plt.ylabel("Im(S)")
                            plt.title(f"Verification IQ Circle - {r['label']} [{status}] (chi^2: {r.get('chisq_fit', np.nan):.4f})")
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            plt.tight_layout()
                            iq_path = os.path.join(target_dir, f"blind_search_verification_IQ_{r['label']}.png")
                            plt.savefig(iq_path, dpi=120)
                            plt.close()
                            print(f"Saved verification IQ plot ({status}) to: {iq_path}")
            except Exception as e:
                print(f"Warning: Failed to generate visualization plots: {e}")
