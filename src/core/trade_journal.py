# --- START OF FILE trade_journal.py ---
"""
Professional Trade Journal Module
================================================================================
Enterprise-grade trade recording with advanced analytics, performance metrics,
trade scoring, pattern analysis, and comprehensive reporting.

Features:
- Complete trade lifecycle tracking with timestamps
- Multi-factor trade scoring (entry, management, exit)
- Win/loss streak analysis and psychology tracking  
- Session and pair performance breakdown
- R:R ratio tracking and optimization suggestions
- Trade pattern recognition and edge detection
- Export to multiple formats (JSON, CSV, HTML reports)
"""

import json
import os
import logging
import hashlib
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import pandas as pd
import pytz

# Configure logging
log = logging.getLogger('TradeJournal')
if not log.handlers:
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(handler)


class TradeOutcome(Enum):
    """Trade outcome classification."""
    OPEN = "OPEN"
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    PARTIAL = "PARTIAL"


class ExitType(Enum):
    """Categorized exit reasons."""
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    TP3_HIT = "TP3_HIT"
    SL_HIT = "SL_HIT"
    TRAILING_STOP = "TRAILING_STOP"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    MANUAL_PROFIT = "MANUAL_PROFIT"
    MANUAL_LOSS = "MANUAL_LOSS"
    TIME_EXIT = "TIME_EXIT"
    NEWS_EXIT = "NEWS_EXIT"
    GUARDIAN_CLOSE = "GUARDIAN_CLOSE"
    EMERGENCY_CLOSE = "EMERGENCY_CLOSE"
    SIGNAL_INVALIDATED = "SIGNAL_INVALIDATED"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"


@dataclass
class TradeScore:
    """Trade quality scoring breakdown."""
    entry_score: float = 0.0       # Quality of entry timing
    management_score: float = 0.0  # How well position was managed
    exit_score: float = 0.0        # Quality of exit timing
    risk_score: float = 0.0        # Risk management adherence
    overall_score: float = 0.0     # Weighted average
    rating: str = "C"              # A+, A, B, C, D, F
    
    def calculate_overall(self) -> None:
        """Calculate overall score from components."""
        weights = {'entry': 0.30, 'management': 0.25, 'exit': 0.25, 'risk': 0.20}
        self.overall_score = (
            self.entry_score * weights['entry'] +
            self.management_score * weights['management'] +
            self.exit_score * weights['exit'] +
            self.risk_score * weights['risk']
        )
        # Convert to letter grade
        if self.overall_score >= 95: self.rating = "A+"
        elif self.overall_score >= 85: self.rating = "A"
        elif self.overall_score >= 75: self.rating = "B"
        elif self.overall_score >= 65: self.rating = "C"
        elif self.overall_score >= 50: self.rating = "D"
        else: self.rating = "F"


@dataclass
class TradeEntry:
    """Complete trade entry record with all metadata."""
    # Identification
    id: str = ""
    ticket: int = 0
    
    # Core Trade Info
    pair: str = ""
    direction: str = ""  # BUY/SELL
    timeframe: str = ""
    
    # Entry Details
    entry_time: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    take_profits: Dict = field(default_factory=dict)  # {tp1: {price, pct}, tp2:...}
    position_size: float = 0.0
    
    # Signal Quality
    grade: str = "C"
    grade_score: float = 0.0
    confidence: float = 0.0
    
    # Confluence Factors
    session_active: List[str] = field(default_factory=list)
    session_quality: float = 0.0
    mtf_alignment: str = ""
    market_structure: str = ""
    trend_strength: str = ""
    orderflow_status: str = ""
    patterns_detected: List[str] = field(default_factory=list)
    divergences: Dict = field(default_factory=dict)
    
    # Entry Reasons (human readable)
    entry_reasons: List[str] = field(default_factory=list)
    entry_score: float = 0.0  # 0-100
    
    # Trade Type
    is_auto_trade: bool = False
    strategy_used: str = ""
    
    # Exit Details (filled on close)
    exit_time: str = ""
    exit_price: float = 0.0
    exit_type: str = ""
    exit_reasons: List[str] = field(default_factory=list)
    
    # Results
    pnl_pips: float = 0.0
    pnl_money: float = 0.0
    pnl_percent: float = 0.0
    rr_achieved: float = 0.0
    max_favorable_excursion: float = 0.0  # MFE in pips
    max_adverse_excursion: float = 0.0    # MAE in pips
    hold_time_minutes: int = 0
    
    # Scoring
    trade_score: Dict = field(default_factory=dict)
    outcome: str = "OPEN"
    status: str = "OPEN"
    
    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class TradeJournal:
    """
    Professional-grade Trade Journal with advanced analytics.
    
    Features:
    - Complete trade lifecycle tracking
    - Performance analytics and pattern detection
    - Win/loss streak analysis
    - Session and pair-based statistics
    - Trade scoring and improvement suggestions
    - Multiple export formats
    """
    
    # Configuration
    JOURNAL_FILE = "data/trade_journal.json"
    ANALYTICS_FILE = "data/trade_analytics.json"
    
    # Trade scoring thresholds
    SCORING_CONFIG = {
        'entry': {
            'grade_weight': 0.4,         # Signal grade importance
            'session_weight': 0.2,        # Trading session timing
            'confluence_weight': 0.25,    # Confluence factors
            'timing_weight': 0.15         # Entry timing precision
        },
        'management': {
            'breakeven_moved': 20,        # Points for moving to BE
            'trailing_used': 20,          # Points for trailing
            'partial_close': 30,          # Points for partial closes
            'hold_optimal': 30            # Points for optimal hold time
        },
        'exit': {
            'tp_hit': 100,                # Full points for hitting TP
            'trailing_profit': 85,        # Good - locked in profit
            'manual_profit': 70,          # Okay - manual profit
            'breakeven': 50,              # Neutral
            'manual_loss': 30,            # Better than SL
            'sl_hit': 20                  # SL hit (expected loss)
        },
        'risk': {
            'proper_sizing': 40,          # Correct position size
            'sl_respected': 30,           # SL was respected
            'rr_positive': 30             # Positive R:R ratio
        }
    }
    
    def __init__(self, journal_file: str = None, timezone: str = 'UTC'):
        """Initialize Professional Trade Journal."""
        self.journal_file = journal_file or self.JOURNAL_FILE
        self.analytics_file = self.ANALYTICS_FILE
        self.timezone = pytz.timezone(timezone)
        
        # Trade storage
        self.entries: Dict[str, TradeEntry] = {}  # id -> TradeEntry
        self.closed_trades: List[Dict] = []
        
        # Analytics cache
        self.streak_data = {'current_streak': 0, 'streak_type': None, 
                           'best_win_streak': 0, 'worst_loss_streak': 0}
        self.session_stats: Dict[str, Dict] = {}
        self.pair_stats: Dict[str, Dict] = {}
        self.daily_stats: Dict[str, Dict] = {}
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.journal_file), exist_ok=True)
        
        # Load existing data
        self._load_journal()
        self._load_analytics()
        
        log.info(f"TradeJournal initialized: {len(self.entries)} open, {len(self.closed_trades)} closed")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    def _load_journal(self) -> None:
        """Load journal from disk."""
        try:
            if os.path.exists(self.journal_file):
                with open(self.journal_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Load open trades
                for entry_data in data.get('open_trades', []):
                    entry = TradeEntry(**entry_data)
                    self.entries[entry.id] = entry
                
                # Load closed trades
                self.closed_trades = data.get('closed_trades', [])
                
                log.info(f"Loaded journal: {len(self.entries)} open, {len(self.closed_trades)} closed")
        except Exception as e:
            log.error(f"Error loading journal: {e}")
    
    def _save_journal(self) -> None:
        """Save journal to disk."""
        try:
            data = {
                'open_trades': [asdict(e) for e in self.entries.values()],
                'closed_trades': self.closed_trades,
                'saved_at': datetime.now(self.timezone).isoformat()
            }
            with open(self.journal_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Error saving journal: {e}")
    
    def _load_analytics(self) -> None:
        """Load analytics cache from disk."""
        try:
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.streak_data = data.get('streak_data', self.streak_data)
                    self.session_stats = data.get('session_stats', {})
                    self.pair_stats = data.get('pair_stats', {})
                    self.daily_stats = data.get('daily_stats', {})
        except Exception as e:
            log.debug(f"Analytics load error: {e}")
    
    def _save_analytics(self) -> None:
        """Save analytics cache to disk."""
        try:
            data = {
                'streak_data': self.streak_data,
                'session_stats': self.session_stats,
                'pair_stats': self.pair_stats,
                'daily_stats': self.daily_stats,
                'saved_at': datetime.now(self.timezone).isoformat()
            }
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.debug(f"Analytics save error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENTRY RECORDING
    # ═══════════════════════════════════════════════════════════════════════════
    def record_entry(self, 
                     ticket: int,
                     signal: Dict,
                     confluence: Dict = None,
                     reasons: List[str] = None,
                     is_auto_trade: bool = False,
                     strategy: str = "default") -> TradeEntry:
        """
        Record a new trade entry with comprehensive details.
        
        Args:
            ticket: MT5 position ticket
            signal: Complete signal dictionary from SignalGenerator
            confluence: Confluence breakdown data
            reasons: Manual entry reasons (auto-generated if None)
            is_auto_trade: Whether this was an automatic trade
            strategy: Strategy name used for this trade
            
        Returns:
            TradeEntry object
        """
        try:
            now = datetime.now(self.timezone)
            
            # Generate unique ID
            trade_id = self._generate_trade_id(ticket, signal.get('pair', ''), now)
            
            # Extract signal data
            session = signal.get('session', {})
            mtf = signal.get('mtf_confluence', {})
            struct = signal.get('market_structure', {})
            
            # Create entry record
            entry = TradeEntry(
                id=trade_id,
                ticket=ticket,
                pair=signal.get('pair', 'UNKNOWN'),
                direction=signal.get('direction', 'UNKNOWN'),
                timeframe=signal.get('timeframe', 'H1'),
                
                # Entry details
                entry_time=now.isoformat(),
                entry_price=signal.get('entry_price', 0),
                stop_loss=signal.get('stop_loss', 0),
                take_profit=signal.get('take_profit', 0),
                take_profits=signal.get('take_profits', {}),
                position_size=signal.get('position_size', 0.01),
                
                # Signal quality
                grade=signal.get('grade', 'C'),
                grade_score=signal.get('grade_score', 0),
                confidence=signal.get('confidence', 0),
                
                # Confluence
                session_active=session.get('active', []),
                session_quality=session.get('quality', 0.5),
                mtf_alignment=mtf.get('alignment', 'UNKNOWN'),
                market_structure=struct.get('structure', 'NEUTRAL'),
                trend_strength=signal.get('trend_strength', 'UNKNOWN'),
                orderflow_status=signal.get('orderflow', {}).get('status', 'NEUTRAL'),
                patterns_detected=signal.get('patterns', {}).get('detected', []),
                divergences=signal.get('divergences', {}),
                
                # Reasons
                entry_reasons=reasons or self._generate_entry_reasons(signal, confluence),
                entry_score=self._calculate_entry_score(signal, confluence),
                
                # Trade type
                is_auto_trade=is_auto_trade,
                strategy_used=strategy,
                
                # Status
                outcome=TradeOutcome.OPEN.value,
                status="OPEN",
                
                # Metadata
                created_at=now.isoformat(),
                updated_at=now.isoformat()
            )
            
            # Store entry
            self.entries[trade_id] = entry
            self._save_journal()
            
            log.info(f"📝 Trade Entry: #{ticket} {entry.direction} {entry.pair} "
                    f"Grade={entry.grade} Score={entry.entry_score:.0f}")
            
            return entry
            
        except Exception as e:
            log.error(f"Record entry error: {e}")
            return TradeEntry()
    
    def _generate_trade_id(self, ticket: int, pair: str, timestamp: datetime) -> str:
        """Generate unique trade ID."""
        data = f"{ticket}_{pair}_{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_entry_reasons(self, signal: Dict, confluence: Dict = None) -> List[str]:
        """Generate comprehensive human-readable entry reasons."""
        reasons = []
        
        # Grade-based
        grade = signal.get('grade', 'C')
        grade_score = signal.get('grade_score', 0)
        if grade in ['A+', 'A']:
            reasons.append(f"✅ Premium signal quality: Grade {grade} ({grade_score:.0%})")
        elif grade == 'B':
            reasons.append(f"⚡ Good signal quality: Grade {grade} ({grade_score:.0%})")
        
        # Session analysis
        session = signal.get('session', {})
        if session.get('is_optimal'):
            active = ', '.join(session.get('active', [])[:2])
            quality = session.get('quality', 0)
            reasons.append(f"🕐 Optimal session: {active} (Quality: {quality:.0%})")
        
        # MTF Confluence
        mtf = signal.get('mtf_confluence', {})
        alignment = mtf.get('alignment', '')
        if 'FULL' in alignment:
            reasons.append("📊 Full multi-timeframe alignment confirmed")
        elif 'PARTIAL' in alignment:
            reasons.append("📈 Partial MTF alignment present")
        
        # Market Structure
        struct = signal.get('market_structure', {})
        if struct.get('is_bos'):
            direction = 'bullish' if 'BULLISH' in struct.get('structure', '') else 'bearish'
            reasons.append(f"🔷 Break of Structure ({direction}) detected")
        if struct.get('is_choch'):
            reasons.append("🔄 Change of Character - potential reversal")
        
        # Trend
        trend = signal.get('trend_strength', '')
        if 'STRONG' in trend:
            reasons.append(f"🔥 Strong trend momentum: {trend}")
        elif 'MODERATE' in trend:
            reasons.append(f"📈 Moderate trend: {trend}")
        
        # Order Flow
        of = signal.get('orderflow', {})
        if of.get('status') == 'CONFIRM':
            of_reasons = of.get('reasons', [])[:1]
            reason_text = of_reasons[0] if of_reasons else "Order flow aligned"
            reasons.append(f"💰 {reason_text}")
        
        # Patterns
        patterns = signal.get('patterns', {}).get('detected', [])
        if patterns:
            top_patterns = patterns[:2]
            reasons.append(f"📊 Patterns: {', '.join(top_patterns)}")
        
        # Divergence
        div = signal.get('divergences', {})
        direction = signal.get('direction', '')
        if div.get('bullish') and direction == 'BUY':
            reasons.append("📉 Bullish divergence supporting entry")
        elif div.get('bearish') and direction == 'SELL':
            reasons.append("📈 Bearish divergence supporting entry")
        
        # Confluence score
        if confluence:
            score = confluence.get('confluence_pct', 0)
            factors = confluence.get('aligned_factors', 0)
            if score >= 75:
                reasons.append(f"🎯 Excellent confluence: {score:.0f}% ({factors} factors)")
            elif score >= 60:
                reasons.append(f"✨ Good confluence: {score:.0f}% ({factors} factors)")
        
        # AI Confidence
        ai_conf = signal.get('ai_confidence', {})
        if ai_conf.get('confidence', 0) >= 0.70:
            reasons.append(f"🤖 High AI confidence: {ai_conf.get('confidence', 0):.0%}")
        
        if not reasons:
            reasons.append("Standard signal criteria met")
        
        return reasons
    
    def _calculate_entry_score(self, signal: Dict, confluence: Dict = None) -> float:
        """Calculate entry quality score (0-100)."""
        score = 0
        config = self.SCORING_CONFIG['entry']
        
        # Grade contribution
        grade_scores = {'A+': 100, 'A': 90, 'B': 75, 'C': 60, 'D': 40}
        grade = signal.get('grade', 'C')
        score += grade_scores.get(grade, 50) * config['grade_weight']
        
        # Session contribution
        session = signal.get('session', {})
        session_quality = session.get('quality', 0.5)
        score += (session_quality * 100) * config['session_weight']
        
        # Confluence contribution
        if confluence:
            conf_pct = confluence.get('confluence_pct', 50)
            score += conf_pct * config['confluence_weight']
        else:
            score += 50 * config['confluence_weight']
        
        # Timing (based on MTF alignment)
        mtf = signal.get('mtf_confluence', {})
        alignment = mtf.get('alignment', '')
        if 'FULL' in alignment:
            score += 100 * config['timing_weight']
        elif 'PARTIAL' in alignment:
            score += 70 * config['timing_weight']
        else:
            score += 40 * config['timing_weight']
        
        return min(100, max(0, score))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXIT RECORDING
    # ═══════════════════════════════════════════════════════════════════════════
    def record_exit(self,
                    ticket: int,
                    exit_price: float,
                    exit_type: str,
                    reasons: List[str] = None,
                    pnl_pips: float = None,
                    pnl_money: float = None,
                    max_favorable: float = None,
                    max_adverse: float = None) -> Dict:
        """
        Record trade exit with complete analysis.
        
        Args:
            ticket: MT5 position ticket
            exit_price: Price at exit
            exit_type: Type of exit (from ExitType enum)
            reasons: Exit reasons
            pnl_pips: Profit/loss in pips
            pnl_money: Profit/loss in money
            max_favorable: Maximum favorable excursion (pips)
            max_adverse: Maximum adverse excursion (pips)
            
        Returns:
            Complete trade record
        """
        try:
            now = datetime.now(self.timezone)
            
            # Find the entry by ticket
            entry = None
            entry_id = None
            for eid, e in self.entries.items():
                if e.ticket == ticket and e.status == "OPEN":
                    entry = e
                    entry_id = eid
                    break
            
            if not entry:
                log.warning(f"No open entry for ticket {ticket}")
                return {}
            
            # Calculate hold time
            entry_time = datetime.fromisoformat(entry.entry_time)
            hold_minutes = int((now - entry_time).total_seconds() / 60)
            
            # Calculate R:R achieved
            if entry.stop_loss > 0:
                risk = abs(entry.entry_price - entry.stop_loss)
                reward = abs(exit_price - entry.entry_price)
                rr_achieved = reward / risk if risk > 0 else 0
            else:
                rr_achieved = 0
            
            # Determine outcome
            if pnl_pips is not None:
                if pnl_pips > 1:
                    outcome = TradeOutcome.WIN.value
                elif pnl_pips < -1:
                    outcome = TradeOutcome.LOSS.value
                else:
                    outcome = TradeOutcome.BREAKEVEN.value
            else:
                outcome = TradeOutcome.WIN.value if 'TP' in exit_type else (
                    TradeOutcome.LOSS.value if 'SL' in exit_type else TradeOutcome.BREAKEVEN.value
                )
            
            # Update entry with exit info
            entry.exit_time = now.isoformat()
            entry.exit_price = exit_price
            entry.exit_type = exit_type
            entry.exit_reasons = reasons or self._generate_exit_reasons(exit_type, pnl_pips)
            entry.pnl_pips = pnl_pips or 0
            entry.pnl_money = pnl_money or 0
            entry.rr_achieved = round(rr_achieved, 2)
            entry.max_favorable_excursion = max_favorable or 0
            entry.max_adverse_excursion = max_adverse or 0
            entry.hold_time_minutes = hold_minutes
            entry.outcome = outcome
            entry.status = "CLOSED"
            entry.updated_at = now.isoformat()
            
            # Calculate trade score
            entry.trade_score = self._calculate_trade_score(entry)
            
            # Move to closed trades
            closed_record = asdict(entry)
            self.closed_trades.append(closed_record)
            del self.entries[entry_id]
            
            # Update analytics
            self._update_analytics(closed_record)
            
            # Save
            self._save_journal()
            self._save_analytics()
            
            outcome_emoji = '✅' if outcome == 'WIN' else ('❌' if outcome == 'LOSS' else '➡️')
            score = entry.trade_score.get('overall_score', 0)
            log.info(f"📝 Trade Exit: #{ticket} {outcome_emoji} {entry.pair} "
                    f"PnL={pnl_pips:.1f} pips, R:R={rr_achieved:.2f}, Score={score:.0f}")
            
            return closed_record
            
        except Exception as e:
            log.error(f"Record exit error: {e}")
            return {}
    
    def _generate_exit_reasons(self, exit_type: str, pnl_pips: float = None) -> List[str]:
        """Generate detailed exit reasons."""
        reasons = []
        
        exit_messages = {
            'TP1_HIT': "🎯 Take Profit 1 reached (1:1 R:R) - Secured initial profit",
            'TP2_HIT': "🎯 Take Profit 2 reached (2:1 R:R) - Extended runner",  
            'TP3_HIT': "🎯 Take Profit 3 reached (3:1 R:R) - Full target achieved",
            'SL_HIT': "🛑 Stop Loss triggered - Risk management executed",
            'TRAILING_STOP': "📈 Trailing stop locked in profit",
            'BREAKEVEN_STOP': "➡️ Breakeven stop - Protected capital",
            'MANUAL_PROFIT': "👤 Manual close in profit - Trader discretion",
            'MANUAL_LOSS': "👤 Manual close at loss - Early cut",
            'TIME_EXIT': "⏰ Time-based exit - Extended hold duration",
            'NEWS_EXIT': "📰 Closed before high-impact news event",
            'GUARDIAN_CLOSE': "🛡️ Guardian emergency close - System protection",
            'EMERGENCY_CLOSE': "🚨 Emergency close - Manual intervention",
            'SIGNAL_INVALIDATED': "❌ Original signal invalidated",
            'DRAWDOWN_LIMIT': "⚠️ Daily drawdown limit reached"
        }
        
        reasons.append(exit_messages.get(exit_type, f"Exit type: {exit_type}"))
        
        if pnl_pips is not None:
            if pnl_pips > 0:
                reasons.append(f"💰 Profit captured: +{pnl_pips:.1f} pips")
            elif pnl_pips < 0:
                reasons.append(f"💸 Loss incurred: {pnl_pips:.1f} pips")
            else:
                reasons.append("➡️ Breakeven trade")
        
        return reasons
    
    def _calculate_trade_score(self, entry: TradeEntry) -> Dict:
        """Calculate comprehensive trade score."""
        score = TradeScore()
        config = self.SCORING_CONFIG
        
        # Entry score (already calculated)
        score.entry_score = entry.entry_score
        
        # Management score
        mgmt_score = 50  # Base
        if entry.exit_type == 'TRAILING_STOP':
            mgmt_score += config['management']['trailing_used']
        if entry.exit_type == 'BREAKEVEN_STOP':
            mgmt_score += config['management']['breakeven_moved']
        if 'TP' in entry.exit_type and entry.exit_type != 'TP3_HIT':
            mgmt_score += config['management']['partial_close']
        score.management_score = min(100, mgmt_score)
        
        # Exit score
        exit_scores = {
            'TP1_HIT': 85, 'TP2_HIT': 95, 'TP3_HIT': 100,
            'TRAILING_STOP': 85, 'BREAKEVEN_STOP': 50,
            'MANUAL_PROFIT': 70, 'MANUAL_LOSS': 30, 'SL_HIT': 20
        }
        score.exit_score = exit_scores.get(entry.exit_type, 50)
        
        # Risk score
        risk_score = 50
        if entry.stop_loss > 0:
            risk_score += config['risk']['sl_respected']
        if entry.rr_achieved > 1:
            risk_score += config['risk']['rr_positive']
        score.risk_score = min(100, risk_score)
        
        # Calculate overall
        score.calculate_overall()
        
        return asdict(score)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYTICS & STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    def _update_analytics(self, trade: Dict) -> None:
        """Update analytics cache with new trade."""
        pair = trade.get('pair', 'UNKNOWN')
        outcome = trade.get('outcome', 'LOSS')
        pnl = trade.get('pnl_pips', 0)
        
        # Update streak
        if outcome == 'WIN':
            if self.streak_data['streak_type'] == 'WIN':
                self.streak_data['current_streak'] += 1
            else:
                self.streak_data['streak_type'] = 'WIN'
                self.streak_data['current_streak'] = 1
            
            if self.streak_data['current_streak'] > self.streak_data['best_win_streak']:
                self.streak_data['best_win_streak'] = self.streak_data['current_streak']
                
        elif outcome == 'LOSS':
            if self.streak_data['streak_type'] == 'LOSS':
                self.streak_data['current_streak'] += 1
            else:
                self.streak_data['streak_type'] = 'LOSS'
                self.streak_data['current_streak'] = 1
            
            if self.streak_data['current_streak'] > self.streak_data['worst_loss_streak']:
                self.streak_data['worst_loss_streak'] = self.streak_data['current_streak']
        
        # Update pair stats
        if pair not in self.pair_stats:
            self.pair_stats[pair] = {'wins': 0, 'losses': 0, 'pnl': 0, 'trades': 0}
        
        self.pair_stats[pair]['trades'] += 1
        self.pair_stats[pair]['pnl'] += pnl
        if outcome == 'WIN':
            self.pair_stats[pair]['wins'] += 1
        elif outcome == 'LOSS':
            self.pair_stats[pair]['losses'] += 1
        
        # Update daily stats
        date_key = datetime.now(self.timezone).strftime('%Y-%m-%d')
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {'wins': 0, 'losses': 0, 'pnl': 0, 'trades': 0}
        
        self.daily_stats[date_key]['trades'] += 1
        self.daily_stats[date_key]['pnl'] += pnl
        if outcome == 'WIN':
            self.daily_stats[date_key]['wins'] += 1
        elif outcome == 'LOSS':
            self.daily_stats[date_key]['losses'] += 1
    
    def get_performance_metrics(self, days: int = 30) -> Dict:
        """
        Get comprehensive performance metrics.
        
        Returns detailed statistics including:
        - Win rate, profit factor, expectancy
        - Average win/loss, R:R stats
        - Streak data, consistency metrics
        - Per-pair and per-session breakdown
        """
        try:
            cutoff = datetime.now(self.timezone) - timedelta(days=days)
            
            trades = [t for t in self.closed_trades 
                     if datetime.fromisoformat(t['exit_time']) >= cutoff]
            
            if not trades:
                return {'total_trades': 0, 'message': 'No trades in period'}
            
            # Basic stats
            wins = [t for t in trades if t['outcome'] == 'WIN']
            losses = [t for t in trades if t['outcome'] == 'LOSS']
            
            total_trades = len(trades)
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            # P&L stats
            pnl_list = [t['pnl_pips'] for t in trades if t.get('pnl_pips') is not None]
            total_pnl = sum(pnl_list)
            avg_pnl = statistics.mean(pnl_list) if pnl_list else 0
            
            win_pnl = [t['pnl_pips'] for t in wins if t.get('pnl_pips')]
            loss_pnl = [abs(t['pnl_pips']) for t in losses if t.get('pnl_pips')]
            
            avg_win = statistics.mean(win_pnl) if win_pnl else 0
            avg_loss = statistics.mean(loss_pnl) if loss_pnl else 0
            
            # Profit factor
            gross_profit = sum(win_pnl)
            gross_loss = sum(loss_pnl)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expectancy
            expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
            
            # R:R stats
            rr_values = [t['rr_achieved'] for t in trades if t.get('rr_achieved')]
            avg_rr = statistics.mean(rr_values) if rr_values else 0
            
            # Score stats
            scores = [t.get('trade_score', {}).get('overall_score', 0) for t in trades]
            avg_score = statistics.mean(scores) if scores else 0
            
            # By grade
            by_grade = {}
            for grade in ['A+', 'A', 'B', 'C']:
                grade_trades = [t for t in trades if t.get('grade') == grade]
                if grade_trades:
                    grade_wins = sum(1 for t in grade_trades if t['outcome'] == 'WIN')
                    by_grade[grade] = {
                        'count': len(grade_trades),
                        'win_rate': grade_wins / len(grade_trades) * 100,
                        'avg_pnl': statistics.mean([t['pnl_pips'] for t in grade_trades if t.get('pnl_pips')])
                    }
            
            # Auto vs Manual
            auto_trades = [t for t in trades if t.get('is_auto_trade')]
            manual_trades = [t for t in trades if not t.get('is_auto_trade')]
            
            return {
                'period_days': days,
                'total_trades': total_trades,
                'wins': win_count,
                'losses': loss_count,
                'win_rate': round(win_rate, 1),
                
                'total_pnl_pips': round(total_pnl, 1),
                'avg_pnl_pips': round(avg_pnl, 1),
                'avg_win_pips': round(avg_win, 1),
                'avg_loss_pips': round(avg_loss, 1),
                
                'profit_factor': round(profit_factor, 2),
                'expectancy': round(expectancy, 2),
                'avg_rr': round(avg_rr, 2),
                
                'avg_trade_score': round(avg_score, 1),
                
                'streak': self.streak_data.copy(),
                'by_grade': by_grade,
                
                'auto_trades': len(auto_trades),
                'manual_trades': len(manual_trades),
                
                'pair_breakdown': self.pair_stats,
                'daily_breakdown': dict(list(self.daily_stats.items())[-7:])  # Last 7 days
            }
            
        except Exception as e:
            log.error(f"Performance metrics error: {e}")
            return {'error': str(e)}
    
    def get_improvement_suggestions(self) -> List[str]:
        """Analyze trades and provide improvement suggestions."""
        suggestions = []
        metrics = self.get_performance_metrics(30)
        
        if metrics.get('total_trades', 0) < 10:
            return ["📊 Need more trades for meaningful analysis (minimum 10)"]
        
        # Win rate analysis
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 40:
            suggestions.append("⚠️ Win rate below 40% - Consider stricter signal filtering")
        elif win_rate > 70:
            suggestions.append("✅ Excellent win rate! Consider increasing position size on A+ signals")
        
        # Profit factor
        pf = metrics.get('profit_factor', 0)
        if pf < 1.0:
            suggestions.append("🔴 Profit factor < 1.0 - Review exit strategy")
        elif pf > 2.0:
            suggestions.append("✅ Strong profit factor - Current strategy is working well")
        
        # R:R analysis
        avg_rr = metrics.get('avg_rr', 0)
        if avg_rr < 1.0:
            suggestions.append("📉 Average R:R < 1.0 - Let winners run longer or cut losses faster")
            
        # Grade analysis
        by_grade = metrics.get('by_grade', {})
        if by_grade:
            best_grade = max(by_grade.keys(), key=lambda g: by_grade[g].get('win_rate', 0))
            suggestions.append(f"📊 Grade {best_grade} performs best - Focus more on these signals")
        
        # Streak warning
        streak = metrics.get('streak', {})
        if streak.get('streak_type') == 'LOSS' and streak.get('current_streak', 0) >= 3:
            suggestions.append("⚠️ On a losing streak - Consider taking a break or reducing size")
        
        return suggestions
    
    def get_trading_edge(self) -> Dict:
        """Calculate trading edge and statistical significance."""
        metrics = self.get_performance_metrics(90)
        
        if metrics.get('total_trades', 0) < 30:
            return {'edge_exists': False, 'reason': 'Insufficient data (need 30+ trades)'}
        
        win_rate = metrics.get('win_rate', 0) / 100
        avg_win = metrics.get('avg_win_pips', 0)
        avg_loss = metrics.get('avg_loss_pips', 1)
        
        # Kelly Criterion
        if avg_loss > 0:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
            kelly_pct = max(0, min(kelly * 100, 25))  # Cap at 25%
        else:
            kelly_pct = 0
        
        edge_exists = metrics.get('profit_factor', 0) > 1.2 and win_rate > 0.45
        
        return {
            'edge_exists': edge_exists,
            'win_rate': round(win_rate * 100, 1),
            'profit_factor': metrics.get('profit_factor', 0),
            'expectancy': metrics.get('expectancy', 0),
            'kelly_percentage': round(kelly_pct, 1),
            'recommended_risk': round(kelly_pct * 0.5, 1),  # Half Kelly (safer)
            'total_trades_analyzed': metrics.get('total_trades', 0)
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    def get_open_trades(self) -> List[Dict]:
        """Get all currently open trades."""
        return [asdict(e) for e in self.entries.values()]
    
    def get_trade_by_ticket(self, ticket: int) -> Optional[Dict]:
        """Get trade record by ticket number."""
        for entry in self.entries.values():
            if entry.ticket == ticket:
                return asdict(entry)
        for trade in self.closed_trades:
            if trade.get('ticket') == ticket:
                return trade
        return None
    
    def get_recent_trades(self, count: int = 20) -> List[Dict]:
        """Get most recent closed trades."""
        return sorted(self.closed_trades, 
                     key=lambda x: x.get('exit_time', ''), 
                     reverse=True)[:count]
    
    def export_to_csv(self, filepath: str = None) -> str:
        """Export all trades to CSV."""
        filepath = filepath or "data/trade_journal_export.csv"
        try:
            df = pd.DataFrame(self.closed_trades)
            df.to_csv(filepath, index=False)
            log.info(f"Exported {len(self.closed_trades)} trades to {filepath}")
            return filepath
        except Exception as e:
            log.error(f"Export error: {e}")
            return ""
    
    def add_note(self, ticket: int, note: str) -> bool:
        """Add a note to a trade."""
        for entry in self.entries.values():
            if entry.ticket == ticket:
                entry.notes = note
                entry.updated_at = datetime.now(self.timezone).isoformat()
                self._save_journal()
                return True
        return False
    
    def add_tags(self, ticket: int, tags: List[str]) -> bool:
        """Add tags to a trade."""
        for entry in self.entries.values():
            if entry.ticket == ticket:
                entry.tags.extend(tags)
                entry.updated_at = datetime.now(self.timezone).isoformat()
                self._save_journal()
                return True
        return False


# --- END OF FILE trade_journal.py ---
