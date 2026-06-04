# --- START OF FILE strategy_ui.py ---
"""
Strategy Control UI
================================================================================
Professional UI panel for managing trading strategies.

Features:
- View all loaded strategies
- Enable/disable strategies with toggle
- View strategy performance metrics
- Deploy strategies to live trading
- Reload strategies without restart
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime

# Import strategy manager
from src.core.strategy_manager import StrategyManager
from src.core.base_strategy import BaseStrategy

# Theme colors (matching app.py)
BG_DARK = "#0a0a0f"
BG_CARD = "#12121a"
BG_HOVER = "#1a1a2e"
BORDER_COLOR = "#2a2a3e"
ACCENT_TEAL = "#00d9a5"
ACCENT_BLUE = "#4a9eff"
SUCCESS_GREEN = "#00c853"
DANGER_RED = "#ff4757"
WARNING_YELLOW = "#ffa502"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#8b8b9e"

# Fonts
HEADER_FONT = ("Roboto Medium", 16)
BOLD_FONT = ("Roboto Medium", 12)
MAIN_FONT = ("Roboto", 12)
SMALL_FONT = ("Roboto", 10)

logger = logging.getLogger("StrategyUI")


class StrategyControlFrame(ctk.CTkFrame):
    """
    Strategy Control Panel for managing trading strategies.
    
    Can be embedded in the main app or used as a standalone tab.
    """
    
    def __init__(self, parent, 
                 strategy_manager: StrategyManager = None,
                 on_deploy_strategy: Callable = None,
                 **kwargs):
        """
        Initialize Strategy Control Panel.
        
        Args:
            parent: Parent widget
            strategy_manager: StrategyManager instance (creates one if None)
            on_deploy_strategy: Callback when strategy is deployed to live
        """
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.strategy_manager = strategy_manager or StrategyManager()
        self.on_deploy_strategy = on_deploy_strategy
        self.strategy_cards: Dict[str, ctk.CTkFrame] = {}
        self.strategy_toggles: Dict[str, ctk.CTkSwitch] = {}
        
        # Build UI
        self._build_ui()
        
        # Load strategies
        self._load_strategies()
    
    def _build_ui(self):
        """Build the strategy control UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        # Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=12)
        
        ctk.CTkLabel(
            title_frame,
            text="🎯 Strategy Control Center",
            font=HEADER_FONT,
            text_color=TEXT_PRIMARY
        ).pack(side="left")
        
        # Action buttons
        btn_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        btn_frame.pack(side="right")
        
        self.reload_btn = ctk.CTkButton(
            btn_frame,
            text="🔄 Reload",
            width=90,
            height=32,
            corner_radius=8,
            fg_color=BG_HOVER,
            hover_color=BORDER_COLOR,
            font=BOLD_FONT,
            command=self._reload_strategies
        )
        self.reload_btn.pack(side="left", padx=5)
        
        self.enable_all_btn = ctk.CTkButton(
            btn_frame,
            text="✅ Enable All",
            width=100,
            height=32,
            corner_radius=8,
            fg_color=SUCCESS_GREEN,
            hover_color="#10b981",
            font=BOLD_FONT,
            command=self._enable_all
        )
        self.enable_all_btn.pack(side="left", padx=5)
        
        self.disable_all_btn = ctk.CTkButton(
            btn_frame,
            text="⏸️ Disable All",
            width=100,
            height=32,
            corner_radius=8,
            fg_color=DANGER_RED,
            hover_color="#dc2626",
            font=BOLD_FONT,
            command=self._disable_all
        )
        self.disable_all_btn.pack(side="left")
        
        # Stats bar
        stats_frame = ctk.CTkFrame(header, fg_color=BG_DARK, corner_radius=8)
        stats_frame.pack(fill="x", padx=15, pady=(0, 12))
        
        self.total_label = ctk.CTkLabel(
            stats_frame,
            text="📊 Total: 0",
            font=MAIN_FONT,
            text_color=TEXT_SECONDARY
        )
        self.total_label.pack(side="left", padx=15, pady=8)
        
        self.active_label = ctk.CTkLabel(
            stats_frame,
            text="✅ Active: 0",
            font=MAIN_FONT,
            text_color=SUCCESS_GREEN
        )
        self.active_label.pack(side="left", padx=15)
        
        self.signals_label = ctk.CTkLabel(
            stats_frame,
            text="📈 Signals Today: 0",
            font=MAIN_FONT,
            text_color=ACCENT_BLUE
        )
        self.signals_label.pack(side="left", padx=15)
        
        # ═══════════════════════════════════════════════════════════════════
        # STRATEGY LIST
        # ═══════════════════════════════════════════════════════════════════
        list_container = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        list_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        list_container.grid_columnconfigure(0, weight=1)
        list_container.grid_rowconfigure(0, weight=1)
        
        # Scrollable frame for strategies
        self.scroll_frame = ctk.CTkScrollableFrame(
            list_container,
            fg_color="transparent",
            scrollbar_button_color=BORDER_COLOR,
            scrollbar_button_hover_color=ACCENT_TEAL
        )
        self.scroll_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.scroll_frame.grid_columnconfigure(0, weight=1)
        
        # Empty state
        self.empty_label = ctk.CTkLabel(
            self.scroll_frame,
            text="No strategies loaded.\nPlace strategy files in /strategies folder and click Reload.",
            font=MAIN_FONT,
            text_color=TEXT_SECONDARY,
            justify="center"
        )
        self.empty_label.grid(row=0, column=0, pady=50)
    
    def _load_strategies(self):
        """Load strategies from folder."""
        try:
            count = self.strategy_manager.load_strategies_from_folder()
            self._refresh_strategy_list()
            logger.info(f"Loaded {count} strategies")
        except Exception as e:
            logger.error(f"Load strategies error: {e}")
    
    def _refresh_strategy_list(self):
        """Refresh the strategy list display."""
        # Clear existing cards
        for card in self.strategy_cards.values():
            card.destroy()
        self.strategy_cards.clear()
        self.strategy_toggles.clear()
        
        # Get strategies
        strategies = self.strategy_manager.get_registered_strategies()
        
        if not strategies:
            self.empty_label.grid()
            self._update_stats()
            return
        
        self.empty_label.grid_remove()
        
        # Create cards for each strategy
        for i, strat_info in enumerate(strategies):
            self._create_strategy_card(i, strat_info)
        
        self._update_stats()
    
    def _create_strategy_card(self, row: int, strat_info: Dict):
        """Create a card for a strategy."""
        name = strat_info.get('name', 'Unknown')
        
        card = ctk.CTkFrame(
            self.scroll_frame,
            fg_color=BG_DARK,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_COLOR
        )
        card.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        card.grid_columnconfigure(1, weight=1)
        
        # ─── Left: Toggle ───
        toggle_frame = ctk.CTkFrame(card, fg_color="transparent", width=80)
        toggle_frame.grid(row=0, column=0, rowspan=2, padx=15, pady=15)
        
        toggle_var = tk.BooleanVar(value=strat_info.get('enabled', True))
        toggle = ctk.CTkSwitch(
            toggle_frame,
            text="",
            variable=toggle_var,
            onvalue=True,
            offvalue=False,
            button_color=SUCCESS_GREEN,
            button_hover_color="#10b981",
            progress_color=SUCCESS_GREEN,
            command=lambda n=name, v=toggle_var: self._toggle_strategy(n, v.get())
        )
        toggle.pack()
        self.strategy_toggles[name] = toggle
        
        # ─── Center: Info ───
        info_frame = ctk.CTkFrame(card, fg_color="transparent")
        info_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=(15, 5))
        
        # Name and type badge
        name_row = ctk.CTkFrame(info_frame, fg_color="transparent")
        name_row.pack(fill="x")
        
        ctk.CTkLabel(
            name_row,
            text=name,
            font=BOLD_FONT,
            text_color=TEXT_PRIMARY
        ).pack(side="left")
        
        # Type badge
        strat_type = strat_info.get('type', 'trend')
        badge_colors = {
            'trend': ACCENT_TEAL,
            'breakout': ACCENT_BLUE,
            'scalp': WARNING_YELLOW,
            'reversal': "#9b59b6"
        }
        
        type_badge = ctk.CTkLabel(
            name_row,
            text=strat_type.upper(),
            font=SMALL_FONT,
            text_color=TEXT_PRIMARY,
            fg_color=badge_colors.get(strat_type, BORDER_COLOR),
            corner_radius=4,
            padx=8,
            pady=2
        )
        type_badge.pack(side="left", padx=10)
        
        # Version
        version = strat_info.get('version', '1.0.0')
        ctk.CTkLabel(
            name_row,
            text=f"v{version}",
            font=SMALL_FONT,
            text_color=TEXT_SECONDARY
        ).pack(side="left")
        
        # Description
        desc = strat_info.get('description', 'No description')[:80]
        ctk.CTkLabel(
            info_frame,
            text=desc,
            font=SMALL_FONT,
            text_color=TEXT_SECONDARY
        ).pack(anchor="w", pady=(5, 0))
        
        # ─── Stats Row ───
        stats_frame = ctk.CTkFrame(card, fg_color="transparent")
        stats_frame.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 15))
        
        stats = strat_info.get('stats', {})
        signals = stats.get('signals_generated', 0)
        trades = stats.get('trades_executed', 0)
        wins = stats.get('wins', 0)
        win_rate = (wins / trades * 100) if trades > 0 else 0
        total_pips = stats.get('total_pips', 0)
        
        stat_items = [
            ("📊 Signals", signals),
            ("💹 Trades", trades),
            ("🎯 Win Rate", f"{win_rate:.0f}%"),
            ("💰 Pips", f"{total_pips:+.1f}")
        ]
        
        for label, value in stat_items:
            stat_label = ctk.CTkLabel(
                stats_frame,
                text=f"{label}: {value}",
                font=SMALL_FONT,
                text_color=TEXT_SECONDARY
            )
            stat_label.pack(side="left", padx=(0, 20))
        
        # ─── Right: Actions ───
        action_frame = ctk.CTkFrame(card, fg_color="transparent")
        action_frame.grid(row=0, column=2, rowspan=2, padx=15, pady=15)
        
        # Deploy button
        deploy_btn = ctk.CTkButton(
            action_frame,
            text="🚀 Deploy",
            width=80,
            height=30,
            corner_radius=6,
            fg_color=ACCENT_TEAL,
            hover_color="#00b894",
            font=BOLD_FONT,
            command=lambda n=name: self._deploy_strategy(n)
        )
        deploy_btn.pack(pady=(0, 5))
        
        # Settings button
        settings_btn = ctk.CTkButton(
            action_frame,
            text="⚙️ Config",
            width=80,
            height=30,
            corner_radius=6,
            fg_color=BG_HOVER,
            hover_color=BORDER_COLOR,
            font=MAIN_FONT,
            command=lambda n=name: self._show_strategy_settings(n)
        )
        settings_btn.pack()
        
        self.strategy_cards[name] = card
    
    def _update_stats(self):
        """Update header statistics."""
        strategies = self.strategy_manager.get_registered_strategies()
        total = len(strategies)
        active = len([s for s in strategies if s.get('enabled', False)])
        
        total_signals = sum(s.get('stats', {}).get('signals_generated', 0) for s in strategies)
        
        self.total_label.configure(text=f"📊 Total: {total}")
        self.active_label.configure(text=f"✅ Active: {active}")
        self.signals_label.configure(text=f"📈 Signals: {total_signals}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACTIONS
    # ═══════════════════════════════════════════════════════════════════════
    def _toggle_strategy(self, name: str, enabled: bool):
        """Toggle strategy enabled state."""
        if enabled:
            self.strategy_manager.enable_strategy(name)
            logger.info(f"✅ Enabled: {name}")
        else:
            self.strategy_manager.disable_strategy(name)
            logger.info(f"⏸️ Disabled: {name}")
        
        self._update_stats()
    
    def _enable_all(self):
        """Enable all strategies."""
        for name in self.strategy_manager.registered_strategies.keys():
            self.strategy_manager.enable_strategy(name)
            if name in self.strategy_toggles:
                self.strategy_toggles[name].select()
        
        self._update_stats()
        logger.info("✅ Enabled all strategies")
    
    def _disable_all(self):
        """Disable all strategies."""
        for name in self.strategy_manager.registered_strategies.keys():
            self.strategy_manager.disable_strategy(name)
            if name in self.strategy_toggles:
                self.strategy_toggles[name].deselect()
        
        self._update_stats()
        logger.info("⏸️ Disabled all strategies")
    
    def _reload_strategies(self):
        """Reload strategies from folder."""
        self.reload_btn.configure(state="disabled", text="Loading...")
        
        def reload_thread():
            try:
                count = self.strategy_manager.load_strategies_from_folder()
                self.after(0, lambda: self._on_reload_complete(count))
            except Exception as e:
                self.after(0, lambda: self._on_reload_error(str(e)))
        
        threading.Thread(target=reload_thread, daemon=True).start()
    
    def _on_reload_complete(self, count: int):
        """Handle reload completion."""
        self.reload_btn.configure(state="normal", text="🔄 Reload")
        self._refresh_strategy_list()
        messagebox.showinfo("Reload Complete", f"Loaded {count} strategies")
    
    def _on_reload_error(self, error: str):
        """Handle reload error."""
        self.reload_btn.configure(state="normal", text="🔄 Reload")
        messagebox.showerror("Reload Error", f"Failed to reload: {error}")
    
    def _deploy_strategy(self, name: str):
        """Deploy strategy to live trading."""
        if not messagebox.askyesno(
            "Deploy Strategy",
            f"Deploy '{name}' for live trading?\n\n"
            "The bot will use this strategy's signals for execution."
        ):
            return
        
        # Enable the strategy
        self.strategy_manager.enable_strategy(name)
        if name in self.strategy_toggles:
            self.strategy_toggles[name].select()
        
        # Call callback if set
        if self.on_deploy_strategy:
            self.on_deploy_strategy(name)
        
        messagebox.showinfo(
            "Strategy Deployed",
            f"✅ '{name}' is now active for live trading!\n\n"
            "Signals will be generated based on this strategy."
        )
        
        logger.info(f"🚀 Deployed strategy: {name}")
    
    def _show_strategy_settings(self, name: str):
        """Show strategy settings dialog."""
        info = self.strategy_manager.get_strategy_info(name)
        if not info:
            return
        
        # Create settings dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Strategy Settings: {name}")
        dialog.geometry("450x500")
        dialog.configure(fg_color=BG_DARK)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        
        # Header
        ctk.CTkLabel(
            dialog,
            text=f"⚙️ {name}",
            font=HEADER_FONT,
            text_color=TEXT_PRIMARY
        ).pack(pady=15)
        
        # Info section
        info_frame = ctk.CTkFrame(dialog, fg_color=BG_CARD, corner_radius=10)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        details = [
            ("Version", info.get('version', '1.0.0')),
            ("Type", info.get('type', 'trend')),
            ("Author", info.get('author', 'Unknown')),
            ("Timeframes", ', '.join(info.get('timeframes', []))),
            ("Pairs", ', '.join(info.get('pairs', [])[:3])),
        ]
        
        for label, value in details:
            row = ctk.CTkFrame(info_frame, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=5)
            ctk.CTkLabel(row, text=f"{label}:", font=MAIN_FONT, text_color=TEXT_SECONDARY, width=100).pack(side="left")
            ctk.CTkLabel(row, text=str(value), font=MAIN_FONT, text_color=TEXT_PRIMARY).pack(side="left")
        
        # Parameters section
        ctk.CTkLabel(
            dialog,
            text="📊 Parameters",
            font=BOLD_FONT,
            text_color=TEXT_PRIMARY
        ).pack(pady=(20, 10), anchor="w", padx=20)
        
        params_frame = ctk.CTkScrollableFrame(dialog, fg_color=BG_CARD, corner_radius=10, height=200)
        params_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        params = info.get('params', {})
        param_entries = {}
        
        for key, value in params.items():
            row = ctk.CTkFrame(params_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(row, text=key, font=MAIN_FONT, text_color=TEXT_SECONDARY, width=150).pack(side="left")
            
            entry = ctk.CTkEntry(row, width=120, font=MAIN_FONT)
            entry.insert(0, str(value))
            entry.pack(side="right")
            param_entries[key] = entry
        
        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=15)
        
        def save_params():
            try:
                new_params = {}
                for key, entry in param_entries.items():
                    val = entry.get()
                    # Try to convert to number
                    try:
                        if '.' in val:
                            new_params[key] = float(val)
                        else:
                            new_params[key] = int(val)
                    except:
                        new_params[key] = val
                
                self.strategy_manager.set_strategy_params(name, new_params)
                messagebox.showinfo("Saved", "Parameters saved successfully!")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=100,
            fg_color=BG_HOVER,
            hover_color=BORDER_COLOR,
            command=dialog.destroy
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="💾 Save",
            width=100,
            fg_color=SUCCESS_GREEN,
            hover_color="#10b981",
            command=save_params
        ).pack(side="right")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC METHODS
    # ═══════════════════════════════════════════════════════════════════════
    def get_strategy_manager(self) -> StrategyManager:
        """Get the strategy manager instance."""
        return self.strategy_manager
    
    def refresh(self):
        """Refresh the display."""
        self._refresh_strategy_list()


# --- END OF FILE strategy_ui.py ---
