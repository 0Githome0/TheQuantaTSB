
import customtkinter as ctk
import pandas as pd
from datetime import datetime
import threading
from tkinter import messagebox
import logging

try:
    from src.core import insider_api, insider_logic
except ImportError:
    # Handle case where imports fail (e.g. running standalone test)
    pass

# Colors (matching app theme)
BG_DARK = "#0f172a"
BG_CARD = "#1e293b" 
BG_HOVER = "#334155"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
SUCCESS_GREEN = "#10b981"
DANGER_RED = "#ef4444" 

# Fonts
HEADER_FONT = ("Roboto Medium", 20)
SUBHEADER_FONT = ("Roboto Medium", 14)
BODY_FONT = ("Roboto", 12)
BOLD_FONT = ("Roboto Bold", 12)

logger = logging.getLogger("InsiderUI")

class InsiderTradingFrame(ctk.CTkFrame):
    """
    UI Module for the Big Company Trade Analyzer.
    """
    def __init__(self, master, execute_callback=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.execute_callback = execute_callback

        # Analysis Data
        self.trades_data = []
        
        # Layout: 2 Columns (List | Details)
        self.grid_columnconfigure(0, weight=1) # List
        self.grid_columnconfigure(1, weight=2) # Details
        self.grid_rowconfigure(0, weight=1)
        
        # --- Left Panel: Trade List ---
        self.list_panel = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=10)
        self.list_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        
        # Header + Refresh
        header_frame = ctk.CTkFrame(self.list_panel, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(header_frame, text="Recent Insider Signals", font=SUBHEADER_FONT, text_color=TEXT_PRIMARY).pack(side="left")
        
        self.refresh_btn = ctk.CTkButton(
            header_frame, 
            text="🔄 Refresh", 
            width=80, 
            height=24,
            fg_color=BG_CARD, 
            command=self.refresh_data
        )
        self.refresh_btn.pack(side="right")
        
        # Scrollable List
        self.trades_scroll = ctk.CTkScrollableFrame(self.list_panel, fg_color="transparent")
        self.trades_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # --- Right Panel: Detail View ---
        self.detail_panel = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10)
        self.detail_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        
        # Initial Empty State
        self.detail_container = ctk.CTkFrame(self.detail_panel, fg_color="transparent")
        self.detail_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(self.detail_container, text="Select a trade to view AI Analysis", font=HEADER_FONT, text_color=TEXT_SECONDARY).pack(expand=True)
        
        # Auto-load on start
        self.after(1000, self.refresh_data)
        
    def refresh_data(self):
        """Fetch new data in background."""
        self.refresh_btn.configure(state="disabled", text="Loading...")
        # Clear list
        for widget in self.trades_scroll.winfo_children():
            widget.destroy()
            
        threading.Thread(target=self._fetch_and_populate, daemon=True).start()
        
    def _fetch_and_populate(self):
        try:
            trades = insider_api.fetch_recent_trades(limit=25)
            self.trades_data = trades
            
            # Update UI on main thread
            self.after(0, lambda: self._update_list_ui(trades))
        except Exception as e:
            logger.error(f"UI Fetch error: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to fetch data: {e}"))
        finally:
            self.after(0, lambda: self.refresh_btn.configure(state="normal", text="🔄 Refresh"))
            
    def _update_list_ui(self, trades):
        if not trades:
            ctk.CTkLabel(self.trades_scroll, text="No recent trades found.", text_color=TEXT_SECONDARY).pack(pady=20)
            return

        for i, trade in enumerate(trades):
            self._create_trade_card(trade, i)
            
    def _create_trade_card(self, trade, index):
        """Card for the left list."""
        c = BG_CARD if index % 2 == 0 else BG_HOVER
        
        card = ctk.CTkFrame(self.trades_scroll, fg_color=c, corner_radius=6, border_width=1, border_color=BG_HOVER)
        card.pack(fill="x", pady=4, padx=4)
        
        # Click event
        card.bind("<Button-1>", lambda e, t=trade: self.show_trade_details(t))
        
        # Layout
        r1 = ctk.CTkFrame(card, fg_color="transparent")
        r1.pack(fill="x", padx=8, pady=(8, 0))
        
        ticker_lbl = ctk.CTkLabel(r1, text=trade['ticker'], font=BOLD_FONT, text_color=ACCENT_BLUE)
        ticker_lbl.pack(side="left")
        ticker_lbl.bind("<Button-1>", lambda e, t=trade: self.show_trade_details(t))
        
        amt = f"${trade['value']:,.0f}"
        val_lbl = ctk.CTkLabel(r1, text=amt, font=BODY_FONT, text_color=SUCCESS_GREEN if trade['transaction_type'] == 'Buy' else DANGER_RED)
        val_lbl.pack(side="right")
        val_lbl.bind("<Button-1>", lambda e, t=trade: self.show_trade_details(t))
        
        r2 = ctk.CTkFrame(card, fg_color="transparent")
        r2.pack(fill="x", padx=8, pady=(0, 8))
        
        insider = trade['insider_name'][:20] + "..." if len(trade['insider_name']) > 20 else trade['insider_name']
        nm_lbl = ctk.CTkLabel(r2, text=f"{insider} ({trade['transaction_type']})", font=ctk.CTkFont(size=10), text_color=TEXT_SECONDARY)
        nm_lbl.pack(side="left")
        nm_lbl.bind("<Button-1>", lambda e, t=trade: self.show_trade_details(t))

    def show_trade_details(self, trade):
        """Render the right panel."""
        # Clear previous
        for w in self.detail_container.winfo_children():
            w.destroy()
            
        # Analysis (fetch context)
        # Show loading...
        loading = ctk.CTkLabel(self.detail_container, text="Running AI Analysis...", font=HEADER_FONT, text_color=ACCENT_BLUE)
        loading.pack(pady=50)
        
        def analyze():
            context = insider_logic.analyze_trade_context(trade['ticker'], trade['trade_date'])
            confidence = insider_logic.calculate_confidence(trade, context)
            rating_text = insider_logic.generate_ai_rating(trade, context, confidence)
            
            self.after(0, lambda: self._render_details(trade, context, confidence, rating_text))
            
        threading.Thread(target=analyze, daemon=True).start()
        
    def _render_details(self, trade, context, confidence, rating_text):
        for w in self.detail_container.winfo_children():
            w.destroy()
            
        # --- Header ---
        top = ctk.CTkFrame(self.detail_container, fg_color="transparent")
        top.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(top, text=f"{trade['ticker']} - {trade['company_name']}", font=HEADER_FONT, text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(top, text=f"Sector: {context.get('sector', 'N/A')} | Price: ${context.get('current_price', 0):.2f}", 
                     font=SUBHEADER_FONT, text_color=TEXT_SECONDARY).pack(anchor="w")
        
        # --- Badges / Tags ---
        tags_frame = ctk.CTkFrame(top, fg_color="transparent")
        tags_frame.pack(anchor="w", pady=(5,0))
        
        def add_tag(text, color):
            t = ctk.CTkFrame(tags_frame, fg_color=color, corner_radius=6)
            t.pack(side="left", padx=(0, 5))
            ctk.CTkLabel(t, text=text, font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_PRIMARY).pack(padx=8, pady=2)

        # Cluster Tag
        c_info = context.get('cluster_info', '')
        if c_info and "No Cluster" not in c_info:
            add_tag(f"👥 {c_info}", ACCENT_BLUE)
            
        # Tech Tag
        t_conf = context.get('technical_confluence', '')
        if t_conf:
            add_tag(f"📊 {t_conf}", SUCCESS_GREEN)
        
        # --- AI Rating ---
        rating_frame = ctk.CTkFrame(self.detail_container, fg_color=BG_DARK, corner_radius=10, border_width=1, border_color=ACCENT_BLUE)
        rating_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(rating_frame, text="AI ANALYSIS", font=ctk.CTkFont(size=10, weight="bold"), text_color=ACCENT_BLUE).pack(anchor="w", padx=15, pady=(10, 0))
        ctk.CTkLabel(rating_frame, text=rating_text, font=("Roboto", 14), text_color=TEXT_PRIMARY, justify="left").pack(anchor="w", padx=15, pady=10)

        # --- Confidence Gauge ---
        ctk.CTkLabel(self.detail_container, text=f"Confidence Score: {confidence}%", font=BOLD_FONT, text_color=TEXT_PRIMARY).pack(anchor="w", pady=(20, 5))
        
        prog = ctk.CTkProgressBar(self.detail_container, height=12, corner_radius=6)
        prog.pack(fill="x")
        prog.set(confidence / 100)
        prog.configure(progress_color=SUCCESS_GREEN if confidence > 70 else (WARNING_YELLOW := "#f59e0b") if confidence > 40 else DANGER_RED)
        
        # --- Trade Specifics ---
        details_grid = ctk.CTkFrame(self.detail_container, fg_color="transparent")
        details_grid.pack(fill="x", pady=20)
        
        def row(label, val):
            f = ctk.CTkFrame(details_grid, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=label, width=120, anchor="w", text_color=TEXT_SECONDARY).pack(side="left")
            ctk.CTkLabel(f, text=val, anchor="w", text_color=TEXT_PRIMARY).pack(side="left")
            
        row("Insider:", f"{trade['insider_name']} ({trade['insider_title']})")
        row("Date Filed:", trade['filed_date'])
        row("Shares Traded:", f"{trade['shares']:,.0f}")
        row("Price per Share:", f"${trade['price']:.2f}")
        row("Total Value:", f"${trade['value']:,.2f}")
        row("Owned After:", f"{trade['owned_after']:,.0f}")
        row("Price Change:", f"{context.get('price_change_since_trade', 0):+.2f}% since trade")

        # --- Action Button ---
        ctk.CTkButton(
            self.detail_container,
            text="Copy Trade / Enter Position",
            height=40,
            font=SUBHEADER_FONT,
            fg_color=ACCENT_BLUE,
            hover_color="#2563eb",
            command=lambda: self.trigger_execution(trade)
        ).pack(fill="x", pady=20, side="bottom") 
        
    def trigger_execution(self, trade):
        """Prepare execution parameters and call main app callback."""
        if not self.execute_callback:
            messagebox.showwarning("Not Connected", "Execution callback not connected.")
            return

        # Confirm before action
        if not messagebox.askyesno("Confirm Trade", f"Do you want to open a {trade['transaction_type']} position on {trade['ticker']}?"):
            return

        # Map to MT5 parameters
        direction = "BUY" if trade['transaction_type'] == "Buy" else "SELL"
        
        # Call the main app's manual execution handler
        # We pass minimal info, let the main app handle entry price etc.
        try:
             # This assumes execute_callback matches _manual_execute_trade signature roughly
             # or we wrap it in a lambda in App.py.
             # Better: We expect callback(pair, direction, volume, sl_mode, tp_mode )
             self.execute_callback(
                 pair=trade['ticker'], 
                 direction=direction, 
                 volume=0.01, # Default starter size
                 use_market_price=True
             )
        except Exception as e:
            logger.error(f"Execution trigger failed: {e}")
            messagebox.showerror("Execution Error", f"Failed to trigger trade: {e}") 
