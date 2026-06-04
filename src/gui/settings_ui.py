
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import json
import logging

# Theme Colors (Consistent with App)
BG_DARK = "#0f172a"
BG_CARD = "#1e293b"
BG_HOVER = "#334155"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
ACCENT_TEAL = "#14b8a6"
BORDER_COLOR = "#334155"

class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, config, on_save_callback):
        super().__init__(parent)
        self.config = config
        self.on_save_callback = on_save_callback
        
        # Window setup
        self.title("Settings")
        self.geometry("500x600")
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)
        self.grab_set() # Modal dialog
        
        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        ctk.CTkLabel(header, text="Configuration", font=("Roboto Medium", 20), text_color=TEXT_PRIMARY).pack(side="left")
        
        # Tab View
        self.tab_view = ctk.CTkTabview(self, fg_color=BG_CARD, corner_radius=10)
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        self.tab_view.add("Account")
        self.tab_view.add("Trading") 
        self.tab_view.add("Execution")
        
        self._build_account_tab()
        self._build_trading_tab()
        self._build_execution_tab()
        
        # Footer Actions
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=20, pady=20)
        
        ctk.CTkButton(footer, text="Cancel", fg_color=BG_HOVER, hover_color=BORDER_COLOR, width=100, command=self.destroy).pack(side="right", padx=10)
        ctk.CTkButton(footer, text="Save Settings", fg_color=ACCENT_TEAL, hover_color="#0d9488", width=120, command=self.save_settings).pack(side="right")
        
    def _build_account_tab(self):
        tab = self.tab_view.tab("Account")
        tab.grid_columnconfigure(1, weight=1)
        
        mt5_cfg = self.config.get("mt5_details", {})
        
        self._add_entry(tab, 0, "Login ID:", mt5_cfg.get("login", ""), "login_entry")
        self._add_entry(tab, 1, "Password:", mt5_cfg.get("password", ""), "password_entry", show="*")
        self._add_entry(tab, 2, "Server:", mt5_cfg.get("server", ""), "server_entry")
        self._add_entry(tab, 3, "MT5 Path (Optional):", mt5_cfg.get("path", ""), "path_entry")
        
        self.auto_connect_var = ctk.BooleanVar(value=mt5_cfg.get("auto_connect_mt5", False))
        ctk.CTkCheckBox(tab, text="Auto-Connect on Startup", variable=self.auto_connect_var, text_color=TEXT_PRIMARY).grid(row=4, column=1, sticky="w", pady=20)

    def _build_trading_tab(self):
        tab = self.tab_view.tab("Trading")
        tab.grid_columnconfigure(1, weight=1)
        
        pairs = ",".join(self.config.get("trading_pairs", []))
        tfs = ",".join(self.config.get("timeframes", []))
        
        self._add_entry(tab, 0, "Trading Pairs (CSV):", pairs, "pairs_entry")
        ctk.CTkLabel(tab, text="Example: EURUSD, GBPUSD, BTCUSD", font=("Roboto", 10), text_color=TEXT_SECONDARY).grid(row=1, column=1, sticky="w", pady=(0, 10))
        
        self._add_entry(tab, 2, "Timeframes (CSV):", tfs, "tfs_entry")
        
        bot_cfg = self.config.get("bot_settings", {})
        self._add_entry(tab, 3, "Data Fetch Limit (Bars):", str(bot_cfg.get("data_fetch_count", 250)), "fetch_limit_entry")
        self._add_entry(tab, 4, "Monitor Interval (sec):", str(bot_cfg.get("monitor_interval_seconds", 10)), "monitor_int_entry")

    def _build_execution_tab(self):
        tab = self.tab_view.tab("Execution")
        tab.grid_columnconfigure(1, weight=1)
        
        exec_cfg = self.config.get("execution_manager", {})
        
        self._add_entry(tab, 0, "Default Lot Size:", str(exec_cfg.get("default_lot_size", 0.01)), "lot_size_entry")
        self._add_entry(tab, 1, "Default Stop Loss (Points):", str(exec_cfg.get("default_risk_pips", 500)), "sl_entry")
        self._add_entry(tab, 2, "Default Take Profit (Points):", str(exec_cfg.get("default_reward_pips", 1000)), "tp_entry")
        self._add_entry(tab, 3, "Max Slippage:", str(exec_cfg.get("max_slippage", 10)), "slippage_entry")

    def _add_entry(self, parent, row, label, value, attr_name, show=None):
        ctk.CTkLabel(parent, text=label, text_color=TEXT_PRIMARY).grid(row=row, column=0, sticky="w", padx=10, pady=10)
        entry = ctk.CTkEntry(parent, width=200, show=show)
        entry.grid(row=row, column=1, sticky="ew", padx=10, pady=10)
        entry.insert(0, str(value))
        setattr(self, attr_name, entry)

    def save_settings(self):
        try:
            # Update Account
            self.config["mt5_details"] = {
                "login": int(self.login_entry.get() or 0),
                "password": self.password_entry.get(),
                "server": self.server_entry.get(),
                "path": self.path_entry.get(),
                "auto_connect_mt5": self.auto_connect_var.get()
            }
            
            # Update Trading
            raw_pairs = self.pairs_entry.get().split(',')
            self.config["trading_pairs"] = [p.strip().upper() for p in raw_pairs if p.strip()]
            
            raw_tfs = self.tfs_entry.get().split(',')
            self.config["timeframes"] = [t.strip().upper() for t in raw_tfs if t.strip()]
            
            self.config["bot_settings"]["data_fetch_count"] = int(self.fetch_limit_entry.get())
            self.config["bot_settings"]["monitor_interval_seconds"] = int(self.monitor_int_entry.get())
            
            # Update Execution
            self.config["execution_manager"] = self.config.get("execution_manager", {})
            self.config["execution_manager"].update({
                "default_lot_size": float(self.lot_size_entry.get()),
                "default_risk_pips": int(self.sl_entry.get()),
                "default_reward_pips": int(self.tp_entry.get()),
                "max_slippage": int(self.slippage_entry.get())
            })
            
            # Callback to App
            if self.on_save_callback:
                self.on_save_callback()
                
            messagebox.showinfo("Success", "Settings saved successfully!\nRestart app for all changes to take effect.")
            self.destroy()
            
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Invalid input format: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
