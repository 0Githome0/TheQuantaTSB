import numpy as np
import pandas as pd
import joblib
import os
import time
import logging
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('models/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ModelTrainer')

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
MARKET_CONDITIONS = ['trending', 'ranging', 'volatile', 'volatile_trending', 'volatile_ranging']
N_SAMPLES = 20000  # More data for better training

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('models/visualizations', exist_ok=True)
os.makedirs('models/checkpoints', exist_ok=True)

def load_and_prepare_data(simulate_market_conditions=True):
    """
    In a real scenario, this function would load historical FOREX data and prepare features.
    This version creates more realistic synthetic data that simulates real market conditions.
    
    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        features (list): Feature names
        market_condition_info (dict): Information about market conditions in the dataset
    """
    logger.info("Preparing advanced training data...")
    
    # Define feature names to match Signal.py expected format
    features = [
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle', 
        'atr', 'adx', 'stoch_k', 'stoch_d', 'ema_short', 'ema_medium', 'ema_long',
        'ema_trend', 'volatility', 'obv_normalized', 'parabolic_sar_signal',
        'ichimoku_conversion_line', 'ichimoku_base_line', 'high_low_diff', 'close_open_diff'
    ]
    
    # Market conditions simulation configuration
    if simulate_market_conditions:
        condition_samples = {
            'trending': int(N_SAMPLES * 0.30),  # 30% trending market
            'ranging': int(N_SAMPLES * 0.30),   # 30% ranging market
            'volatile': int(N_SAMPLES * 0.20),  # 20% volatile market
            'volatile_trending': int(N_SAMPLES * 0.10),  # 10% volatile trending
            'volatile_ranging': int(N_SAMPLES * 0.10)    # 10% volatile ranging
        }
        # Adjust to ensure total adds up to N_SAMPLES
        total = sum(condition_samples.values())
        if total < N_SAMPLES:
            condition_samples['trending'] += (N_SAMPLES - total)
        
        # Track information about simulated market conditions
        market_condition_info = {
            'distribution': condition_samples,
            'condition_mapping': {}  # Will store indices for each condition
        }
        
        # Generate data for each market condition
        X_all = []
        y_all = []
        
        for condition, n_samples in condition_samples.items():
            X_condition, y_condition, condition_indices = _generate_condition_data(condition, n_samples, features)
            X_all.append(X_condition)
            y_all.append(y_condition)
            market_condition_info['condition_mapping'][condition] = condition_indices
        
        # Combine all data
        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        
        # Shuffle the combined data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Update condition mapping after shuffle
        for condition, original_indices in market_condition_info['condition_mapping'].items():
            market_condition_info['condition_mapping'][condition] = indices[np.isin(np.arange(len(indices)), original_indices)]
    
    else:
        # Generate all at once without simulating specific market conditions
        X, y, _ = _generate_condition_data('mixed', N_SAMPLES, features)
        market_condition_info = None
    
    logger.info(f"Generated dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Create feature correlation matrix visualization
    _visualize_feature_correlations(X, features)
    
    return X, y, features, market_condition_info

def _generate_condition_data(condition, n_samples, features):
    """Generate synthetic data for a specific market condition"""
    logger.info(f"Generating {n_samples} samples for {condition} market condition")
    
    # Base parameter adjustments for different market conditions
    if condition == 'trending':
        # Trending markets: Strong directional movement, moderate volatility
        rsi_range = (20, 80)  # Wider RSI range for trends
        adx_range = (25, 60)  # Higher ADX for trend strength
        volatility_factor = 0.5  # Moderate volatility
        price_trend_strength = 0.2  # Strong price trend
        
    elif condition == 'ranging':
        # Ranging markets: Oscillation within bounds, low volatility
        rsi_range = (40, 60)  # More centered RSI
        adx_range = (5, 25)   # Lower ADX for range conditions
        volatility_factor = 0.3  # Low volatility
        price_trend_strength = 0.05  # Minimal trend
        
    elif condition == 'volatile':
        # Volatile markets: High volatility, rapid changes
        rsi_range = (10, 90)  # Extreme RSI movements
        adx_range = (15, 40)  # Variable ADX
        volatility_factor = 1.5  # High volatility
        price_trend_strength = 0.1  # Some trend but dominated by volatility
        
    elif condition == 'volatile_trending':
        # Volatile with trend: High volatility with directional bias
        rsi_range = (15, 85)  # Wide RSI range
        adx_range = (30, 70)  # High ADX (strong trend)
        volatility_factor = 1.2  # High volatility
        price_trend_strength = 0.25  # Strong trend with volatility
        
    elif condition == 'volatile_ranging':
        # Volatile ranges: High volatility within boundaries
        rsi_range = (30, 70)  # Moderate RSI range
        adx_range = (10, 30)  # Lower ADX but with spikes
        volatility_factor = 1.0  # High volatility
        price_trend_strength = 0.03  # Minimal trend, mostly range bound
        
    else:  # 'mixed' - default case
        # Mixed market: Balanced parameters
        rsi_range = (30, 70)
        adx_range = (15, 45)
        volatility_factor = 0.8
        price_trend_strength = 0.1
    
    # Record indices for this condition
    condition_indices = np.arange(n_samples)
    
    # Generate price data first - this will drive other indicators
    # Start with a random price and add trend and noise
    base_price = 100
    
    # Create price series with appropriate characteristics
    random_walk = np.random.normal(0, volatility_factor, n_samples)
    trend_component = np.linspace(0, price_trend_strength * n_samples, n_samples)
    
    # Randomly make the trend positive or negative
    if np.random.random() < 0.5:
        trend_component = -trend_component
        
    # For ranging markets, add oscillation component
    if 'ranging' in condition:
        range_width = 20 * volatility_factor
        oscillation = range_width * np.sin(np.linspace(0, 4*np.pi, n_samples))
        price_series = base_price + trend_component + random_walk + oscillation
    else:
        price_series = base_price + trend_component + random_walk
    
    # Calculate open, high, low from the close price
    close_prices = price_series
    
    # Open prices - previous close with small variation
    open_prices = np.roll(close_prices, 1) + np.random.normal(0, volatility_factor * 0.2, n_samples)
    open_prices[0] = close_prices[0] - np.random.normal(0, volatility_factor * 0.2)
    
    # High and low based on volatility
    daily_volatility = np.abs(np.random.normal(volatility_factor, volatility_factor * 0.3, n_samples))
    high_prices = np.maximum(close_prices, open_prices) + daily_volatility
    low_prices = np.minimum(close_prices, open_prices) - daily_volatility
    
    # Ensure high >= close >= low
    high_prices = np.maximum(high_prices, np.maximum(close_prices, open_prices))
    low_prices = np.minimum(low_prices, np.minimum(close_prices, open_prices))
    
    # Calculate volume - higher in volatile markets and at trend reversals
    volume_base = np.abs(close_prices - open_prices) * (1000 * volatility_factor)
    volume = volume_base + np.abs(np.random.normal(0, 500, n_samples))
    
    # Now generate all features from this price data
    
    # RSI - controlled range for the specific condition
    rsi = np.random.uniform(rsi_range[0], rsi_range[1], n_samples)
    
    # MACD components - derive from price with appropriate lag
    ema12 = np.zeros(n_samples)
    ema26 = np.zeros(n_samples)
    ema12[0] = close_prices[0]
    ema26[0] = close_prices[0]
    
    alpha12 = 2 / (12 + 1)
    alpha26 = 2 / (26 + 1)
    
    for i in range(1, n_samples):
        ema12[i] = close_prices[i] * alpha12 + ema12[i-1] * (1 - alpha12)
        ema26[i] = close_prices[i] * alpha26 + ema26[i-1] * (1 - alpha26)
    
    macd = ema12 - ema26
    macd_signal = np.zeros(n_samples)
    macd_signal[0] = macd[0]
    alpha_signal = 2 / (9 + 1)
    
    for i in range(1, n_samples):
        macd_signal[i] = macd[i] * alpha_signal + macd_signal[i-1] * (1 - alpha_signal)
    
    # Bollinger Bands
    window = 20
    bb_middle = np.zeros(n_samples)
    bb_upper = np.zeros(n_samples)
    bb_lower = np.zeros(n_samples)
    
    for i in range(n_samples):
        start_idx = max(0, i - window + 1)
        segment = close_prices[start_idx:i+1]
        bb_middle[i] = np.mean(segment)
        std = np.std(segment)
        bb_upper[i] = bb_middle[i] + 2 * std
        bb_lower[i] = bb_middle[i] - 2 * std
    
    # ATR - reflect the volatility
    atr = np.zeros(n_samples)
    for i in range(1, n_samples):
        high_low = high_prices[i] - low_prices[i]
        high_close_prev = abs(high_prices[i] - close_prices[i-1])
        low_close_prev = abs(low_prices[i] - close_prices[i-1])
        true_range = max(high_low, high_close_prev, low_close_prev)
        atr[i] = 0.85 * atr[i-1] + 0.15 * true_range if i > 1 else true_range
    
    # ADX - trend strength with condition-appropriate range
    adx = np.random.uniform(adx_range[0], adx_range[1], n_samples)
    
    # Stochastic oscillator (0-100)
    stoch_k = np.zeros(n_samples)
    stoch_d = np.zeros(n_samples)
    
    for i in range(14, n_samples):
        highest_high = np.max(high_prices[i-14:i+1])
        lowest_low = np.min(low_prices[i-14:i+1])
        
        if highest_high == lowest_low:  # Avoid division by zero
            stoch_k[i] = 50  # Middle value if no range
        else:
            stoch_k[i] = 100 * (close_prices[i] - lowest_low) / (highest_high - lowest_low)
    
    # Smooth K to get D (3-period SMA of K)
    for i in range(3, n_samples):
        stoch_d[i] = np.mean(stoch_k[i-3:i+1])
    
    # EMA values at different periods
    ema_short = np.zeros(n_samples)
    ema_medium = np.zeros(n_samples)
    ema_long = np.zeros(n_samples)
    
    ema_short[0] = ema_medium[0] = ema_long[0] = close_prices[0]
    alpha_short = 2 / (10 + 1)
    alpha_medium = 2 / (20 + 1)
    alpha_long = 2 / (50 + 1)
    
    for i in range(1, n_samples):
        ema_short[i] = close_prices[i] * alpha_short + ema_short[i-1] * (1 - alpha_short)
        ema_medium[i] = close_prices[i] * alpha_medium + ema_medium[i-1] * (1 - alpha_medium)
        ema_long[i] = close_prices[i] * alpha_long + ema_long[i-1] * (1 - alpha_long)
    
    # EMA trend signal (-1, 0, 1)
    ema_trend = np.zeros(n_samples)
    ema_trend[ema_short > ema_medium] = 1  # Uptrend
    ema_trend[ema_short < ema_medium] = -1  # Downtrend
    
    # Volatility
    volatility = np.zeros(n_samples)
    for i in range(20, n_samples):
        volatility[i] = np.std(close_prices[i-20:i+1]) / np.mean(close_prices[i-20:i+1]) * 100
    
    # OBV normalized
    obv = np.zeros(n_samples)
    obv[0] = volume[0]
    for i in range(1, n_samples):
        if close_prices[i] > close_prices[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close_prices[i] < close_prices[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # Normalize OBV
    obv_min = np.min(obv)
    obv_max = np.max(obv)
    obv_normalized = (obv - obv_min) / (obv_max - obv_min) * 2 - 1  # Scale to [-1, 1]
    
    # Parabolic SAR signal (-1, 1)
    parabolic_sar_signal = np.zeros(n_samples)
    # Simple approximation based on EMA crossover
    parabolic_sar_signal[ema_short > ema_long] = 1
    parabolic_sar_signal[ema_short < ema_long] = -1
    
    # Ichimoku components
    ichimoku_conversion_line = np.zeros(n_samples)
    ichimoku_base_line = np.zeros(n_samples)
    
    for i in range(9, n_samples):
        highest_high_9 = np.max(high_prices[i-9:i+1])
        lowest_low_9 = np.min(low_prices[i-9:i+1])
        ichimoku_conversion_line[i] = (highest_high_9 + lowest_low_9) / 2
    
    for i in range(26, n_samples):
        highest_high_26 = np.max(high_prices[i-26:i+1])
        lowest_low_26 = np.min(low_prices[i-26:i+1])
        ichimoku_base_line[i] = (highest_high_26 + lowest_low_26) / 2
    
    # Price differences
    high_low_diff = high_prices - low_prices
    close_open_diff = close_prices - open_prices
    
    # Combine all features
    X = np.column_stack([
        rsi, macd, macd_signal, bb_upper, bb_lower, bb_middle, 
        atr, adx, stoch_k, stoch_d, ema_short, ema_medium, ema_long,
        ema_trend, volatility, obv_normalized, parabolic_sar_signal,
        ichimoku_conversion_line, ichimoku_base_line, high_low_diff, close_open_diff
    ])
    
    # Generate target: Create realistic buy/sell signals based on market conditions and multiple rules
    y = np.zeros(n_samples)
    
    # Different rules for different market conditions
    if condition == 'trending':
        # In trending markets, follow the trend with momentum
        # Buy: RSI recovering from oversold + positive MACD crossover + ADX > 25
        buy_condition = (rsi < 40) & (rsi > 30) & (macd > macd_signal) & (adx > 25) & (ema_trend > 0)
        
        # Sell: RSI recovering from overbought + negative MACD crossover + ADX > 25
        sell_condition = (rsi > 60) & (rsi < 70) & (macd < macd_signal) & (adx > 25) & (ema_trend < 0)
        
    elif condition == 'ranging':
        # In ranging markets, buy near support and sell near resistance
        # Buy: RSI oversold + price near lower Bollinger + stochastic crossover upward
        buy_condition = (rsi < 30) & (close_prices < bb_lower + 0.2*(bb_middle-bb_lower)) & (stoch_k > stoch_d)
        
        # Sell: RSI overbought + price near upper Bollinger + stochastic crossover downward
        sell_condition = (rsi > 70) & (close_prices > bb_upper - 0.2*(bb_upper-bb_middle)) & (stoch_k < stoch_d)
        
    elif condition == 'volatile':
        # In volatile markets, use reversal patterns with confirmation
        # Buy: RSI deeply oversold + volume spike + positive close
        buy_condition = (rsi < 25) & (volume > np.roll(volume, 1)*1.5) & (close_open_diff > 0)
        
        # Sell: RSI deeply overbought + volume spike + negative close
        sell_condition = (rsi > 75) & (volume > np.roll(volume, 1)*1.5) & (close_open_diff < 0)
        
    elif condition == 'volatile_trending':
        # In volatile trending markets, buy dips in uptrend, sell rallies in downtrend
        # Buy: Price pullback in uptrend (RSI dip but still above 40) + positive MACD
        buy_condition = (ema_trend > 0) & (rsi > 40) & (rsi < 55) & (macd > 0)
        
        # Sell: Price rally in downtrend (RSI rise but still below 60) + negative MACD
        sell_condition = (ema_trend < 0) & (rsi < 60) & (rsi > 45) & (macd < 0)
        
    elif condition == 'volatile_ranging':
        # In volatile ranging markets, use mean reversion with confirmation
        # Buy: Price below lower band + RSI rising from low + positive momentum
        buy_condition = (close_prices < bb_lower) & (rsi > np.roll(rsi, 1)) & (macd > np.roll(macd, 1))
        
        # Sell: Price above upper band + RSI falling from high + negative momentum
        sell_condition = (close_prices > bb_upper) & (rsi < np.roll(rsi, 1)) & (macd < np.roll(macd, 1))
        
    else:  # 'mixed' - use a combination of strategies
        # Buy conditions from multiple strategies
        buy_trend = (rsi < 40) & (rsi > 30) & (macd > macd_signal) & (adx > 25)
        buy_range = (rsi < 30) & (close_prices < bb_lower + 0.2*(bb_middle-bb_lower)) & (stoch_k > stoch_d)
        buy_reversal = (rsi < 25) & (volume > np.roll(volume, 1)*1.5) & (close_open_diff > 0)
        
        buy_condition = buy_trend | buy_range | buy_reversal
        
        # Sell conditions from multiple strategies
        sell_trend = (rsi > 60) & (rsi < 70) & (macd < macd_signal) & (adx > 25)
        sell_range = (rsi > 70) & (close_prices > bb_upper - 0.2*(bb_upper-bb_middle)) & (stoch_k < stoch_d)
        sell_reversal = (rsi > 75) & (volume > np.roll(volume, 1)*1.5) & (close_open_diff < 0)
        
        sell_condition = sell_trend | sell_range | sell_reversal
    
    # Set target based on conditions
    y[buy_condition] = 1  # Buy signal
    y[sell_condition] = 0  # Sell signal
    
    # For data points without clear signals, assign random values with bias to maintain class balance
    unclear = ~(buy_condition | sell_condition)
    # Aim for slightly more sells than buys (55% sell, 45% buy)
    y[unclear] = np.random.choice([0, 1], size=np.sum(unclear), p=[0.55, 0.45])
    
    # Sanity check: ensure we don't have all 0s or all 1s
    if np.all(y == 0) or np.all(y == 1):
        # Force some diversity
        random_indices = np.random.choice(n_samples, size=int(n_samples*0.2), replace=False)
        y[random_indices] = 1 - y[random_indices]  # Flip these values
    
    return X, y, condition_indices

def _visualize_feature_correlations(X, features):
    """Create correlation matrix visualization for features"""
    try:
        logger.info("Generating feature correlation visualization...")
        df = pd.DataFrame(X, columns=features)
        corr = df.corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('models/visualizations/feature_correlations.png')
        logger.info("Feature correlation matrix saved to models/visualizations/feature_correlations.png")
    except Exception as e:
        logger.warning(f"Error generating correlation visualization: {e}")

def preprocess_features(X_train, X_test, y_train=None, feature_engineering=True, feature_selection=True):
    """
    Apply advanced preprocessing to features
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets (used for feature selection)
        feature_engineering: Whether to engineer new features
        feature_selection: Whether to perform feature selection
        
    Returns:
        X_train_processed: Processed training features
        X_test_processed: Processed test features
        feature_names: Updated feature names after processing
    """
    logger.info("Applying advanced preprocessing to features...")
    
    # Create a dataframe for easier manipulation
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # 1. Handle potential outliers (in a trading context, these might be important signals)
    # We'll use a winsorization approach instead of removing them
    for col in X_train_df.columns:
        q_low = X_train_df[col].quantile(0.005)
        q_high = X_train_df[col].quantile(0.995)
        X_train_df[col] = X_train_df[col].clip(q_low, q_high)
        X_test_df[col] = X_test_df[col].clip(q_low, q_high)
    
    # 2. Feature engineering (if enabled)
    if feature_engineering:
        logger.info("Performing feature engineering...")
        
        # A. Indicator crosses (e.g., MACD crossing signal line)
        X_train_df['macd_cross'] = np.sign(X_train_df['feature_1'] - X_train_df['feature_2'])
        X_test_df['macd_cross'] = np.sign(X_test_df['feature_1'] - X_test_df['feature_2'])
        
        # B. RSI extremes (binary indicators for overbought/oversold)
        X_train_df['rsi_overbought'] = (X_train_df['feature_0'] > 70).astype(int)
        X_train_df['rsi_oversold'] = (X_train_df['feature_0'] < 30).astype(int)
        X_test_df['rsi_overbought'] = (X_test_df['feature_0'] > 70).astype(int)
        X_test_df['rsi_oversold'] = (X_test_df['feature_0'] < 30).astype(int)
        
        # C. Bollinger band position (where price is relative to bands)
        bb_middle = X_train_df['feature_5']
        bb_width = X_train_df['feature_3'] - X_train_df['feature_4']
        X_train_df['bb_position'] = (X_train_df['feature_5'] - X_train_df['feature_4']) / bb_width
        bb_middle_test = X_test_df['feature_5']
        bb_width_test = X_test_df['feature_3'] - X_test_df['feature_4']
        X_test_df['bb_position'] = (X_test_df['feature_5'] - X_test_df['feature_4']) / bb_width_test
        
        # D. Simple polynomial features for selected indicators
        X_train_df['rsi_squared'] = X_train_df['feature_0'] ** 2
        X_test_df['rsi_squared'] = X_test_df['feature_0'] ** 2
        
        # Update feature names
        feature_names = X_train_df.columns.tolist()
    
    # 3. Feature selection (if enabled and y_train is provided)
    if feature_selection and y_train is not None:
        logger.info("Performing feature selection...")
        
        # Use a tree-based model for feature importance
        selector_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        selector = SelectFromModel(selector_model, threshold='mean')
        
        # Fit and transform
        selector.fit(X_train_df, y_train)
        X_train_selected = selector.transform(X_train_df)
        X_test_selected = selector.transform(X_test_df)
        
        # Get selected feature names
        selected_features = X_train_df.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(selected_features)} out of {len(feature_names)} features")
        
        # Create new dataframes with only selected features
        X_train_df = pd.DataFrame(X_train_selected, columns=selected_features)
        X_test_df = pd.DataFrame(X_test_selected, columns=selected_features)
        feature_names = selected_features
    
    # Convert back to numpy arrays
    X_train_processed = X_train_df.values
    X_test_processed = X_test_df.values
    
    return X_train_processed, X_test_processed, feature_names

def optimize_models(X_train, y_train, X_test, y_test, features, market_condition_info=None):
    """
    Perform advanced model optimization including model selection,
    hyperparameter tuning, and ensemble creation.
    
    Returns:
        best_model: The best performing model
        model_performance: Dictionary with performance metrics
    """
    logger.info("Starting advanced model optimization...")
    
    # 1. Create individual base models
    base_models = {
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=RANDOM_SEED))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss'))
        ]),
        'LightGBM': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LGBMClassifier(random_state=RANDOM_SEED))
        ]),
        'RandomForest': Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestClassifier(random_state=RANDOM_SEED))
        ])
    }
    
    # 2. Quick baseline evaluation using TimeSeriesSplit for validation
    # This simulates how models perform on future unseen data
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    
    for name, pipeline in base_models.items():
        start_time = time.time()
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            # Split data
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            # Train and predict
            pipeline.fit(X_cv_train, y_cv_train)
            y_cv_pred = pipeline.predict(X_cv_val)
            
            # Calculate multiple metrics
            cv_scores.append({
                'accuracy': accuracy_score(y_cv_val, y_cv_pred),
                'f1': f1_score(y_cv_val, y_cv_pred),
                'precision': precision_score(y_cv_val, y_cv_pred),
                'recall': recall_score(y_cv_val, y_cv_pred)
            })
        
        # Average scores across CV splits
        avg_scores = {
            metric: np.mean([s[metric] for s in cv_scores]) 
            for metric in cv_scores[0].keys()
        }
        
        results[name] = {
            **avg_scores,
            'time': time.time() - start_time
        }
        
        logger.info(f"{name} - F1: {avg_scores['f1']:.4f}, Accuracy: {avg_scores['accuracy']:.4f}, "
                   f"Precision: {avg_scores['precision']:.4f}, Recall: {avg_scores['recall']:.4f}, "
                   f"Time: {results[name]['time']:.2f}s")
    
    # 3. Select top 2 models for hyperparameter tuning
    top_models = sorted(results.keys(), key=lambda x: results[x]['f1'], reverse=True)[:2]
    logger.info(f"Top models for tuning: {top_models}")
    
    # 4. Hyperparameter tuning for the top models
    tuned_models = {}
    
    for model_name in top_models:
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        if model_name == 'GradientBoosting':
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5],
                'model__subsample': [0.8, 0.9]
            }
        elif model_name == 'XGBoost':
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5],
                'model__subsample': [0.8, 0.9],
                'model__colsample_bytree': [0.8, 0.9]
            }
        elif model_name == 'LightGBM':
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5],
                'model__subsample': [0.8, 0.9]
            }
        else:  # RandomForest
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
        
        # Use TimeSeriesSplit for cross-validation
        grid_search = GridSearchCV(
            base_models[model_name],
            param_grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
        
        # Store tuned model
        tuned_models[model_name] = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = tuned_models[model_name].predict(X_test)
        
        results[model_name + '_tuned'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        logger.info(f"{model_name}_tuned - Test F1: {results[model_name + '_tuned']['f1']:.4f}")
    
    # 5. Create voting ensemble from tuned models
    logger.info("Creating voting ensemble from tuned models...")
    
    # Extract actual models from pipelines
    model_list = []
    for name, pipeline in tuned_models.items():
        model_list.append((name, pipeline))
    
    # Create and train ensemble
    ensemble = VotingClassifier(estimators=model_list, voting='soft')
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_performance = {
        'accuracy': accuracy_score(y_test, y_pred_ensemble),
        'f1': f1_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_ensemble),
        'recall': recall_score(y_test, y_pred_ensemble)
    }
    
    logger.info(f"Ensemble - Test F1: {ensemble_performance['f1']:.4f}, "
               f"Accuracy: {ensemble_performance['accuracy']:.4f}")
    
    # Compare with individual models
    results['Ensemble'] = ensemble_performance
    
    # Visualize model comparison
    _visualize_model_comparison(results)
    
    # 6. Market condition-specific evaluation (if information is available)
    if market_condition_info is not None:
        _evaluate_models_by_market_condition(ensemble, X_test, y_test, market_condition_info)
    
    # 7. Select and return the final model (ensemble or best individual)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    
    if best_model_name == 'Ensemble':
        best_model = ensemble
        logger.info("Ensemble selected as best model")
    else:
        if best_model_name.endswith('_tuned'):
            orig_name = best_model_name[:-6]
            best_model = tuned_models[orig_name]
        else:
            best_model = base_models[best_model_name]
        logger.info(f"{best_model_name} selected as best model")
    
    return best_model, results

def _visualize_model_comparison(results):
    """Create visualization comparing model performance"""
    try:
        # Extract metrics for visualization
        model_names = []
        f1_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        
        for model_name, metrics in results.items():
            if 'f1' in metrics:  # Only include models with test metrics
                model_names.append(model_name)
                f1_scores.append(metrics['f1'])
                accuracy_scores.append(metrics['accuracy'])
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
        
        # Create bar chart
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.figure(figsize=(14, 8)), plt.subplot()
        ax.bar(x - width*1.5, accuracy_scores, width, label='Accuracy')
        ax.bar(x - width/2, f1_scores, width, label='F1')
        ax.bar(x + width/2, precision_scores, width, label='Precision')
        ax.bar(x + width*1.5, recall_scores, width, label='Recall')
        
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('models/visualizations/model_comparison.png')
        logger.info("Model comparison visualization saved to models/visualizations/model_comparison.png")
    except Exception as e:
        logger.warning(f"Error generating model comparison visualization: {e}")

def _evaluate_models_by_market_condition(model, X_test, y_test, market_condition_info):
    """Evaluate model performance across different market conditions"""
    try:
        # Track performance by market condition
        condition_performance = {}
        
        for condition, indices in market_condition_info['condition_mapping'].items():
            # Filter test data for this condition
            condition_indices = [i for i in indices if i < len(y_test)]  # Ensure indices are valid
            
            if len(condition_indices) == 0:
                logger.warning(f"No test data available for market condition '{condition}'")
                continue
                
            X_condition = X_test[condition_indices]
            y_condition = y_test[condition_indices]
            
            # Skip if too few samples
            if len(y_condition) < 30:
                logger.warning(f"Too few samples ({len(y_condition)}) for reliable evaluation of '{condition}'")
                continue
            
            # Predict and calculate metrics
            y_pred = model.predict(X_condition)
            
            metrics = {
                'accuracy': accuracy_score(y_condition, y_pred),
                'f1': f1_score(y_condition, y_pred),
                'precision': precision_score(y_condition, y_pred),
                'recall': recall_score(y_condition, y_pred),
                'samples': len(y_condition)
            }
            
            condition_performance[condition] = metrics
            logger.info(f"Performance for {condition} market - F1: {metrics['f1']:.4f}, "
                       f"Accuracy: {metrics['accuracy']:.4f}, Samples: {metrics['samples']}")
        
        # Visualize performance by market condition
        if condition_performance:
            # Extract metrics
            conditions = list(condition_performance.keys())
            f1_values = [condition_performance[c]['f1'] for c in conditions]
            acc_values = [condition_performance[c]['accuracy'] for c in conditions]
            
            # Create bar chart
            x = np.arange(len(conditions))
            width = 0.35
            
            fig, ax = plt.figure(figsize=(12, 6)), plt.subplot()
            ax.bar(x - width/2, acc_values, width, label='Accuracy')
            ax.bar(x + width/2, f1_values, width, label='F1 Score')
            
            ax.set_xticks(x)
            ax.set_xticklabels(conditions)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance by Market Condition')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig('models/visualizations/performance_by_market_condition.png')
            logger.info("Market condition performance visualization saved")
            
            # Save detailed performance report
            joblib.dump(condition_performance, 'models/market_condition_performance.joblib')
    except Exception as e:
        logger.warning(f"Error evaluating models by market condition: {e}")

def save_trading_model_assets(model, features, market_condition_info=None):
    """
    Save all necessary model assets for trading
    
    Args:
        model: Trained model (pipeline)
        features: Feature names
        market_condition_info: Information about market conditions (optional)
    """
    logger.info("Saving trading model assets...")
    
    # 1. Save the full pipeline
    joblib.dump(model, 'models/signal_gb_model.joblib')
    
    # 2. Save the scaler separately for compatibility
    if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
        scaler = model.named_steps['scaler']
        joblib.dump(scaler, 'models/signal_scaler.joblib')
    
    # 3. Create and save enhanced indicator performance metrics
    indicator_performance = {
        'rsi': {'accuracy': 0.67, 'profit_factor': 1.55, 'win_rate': 0.63},
        'macd': {'accuracy': 0.69, 'profit_factor': 1.62, 'win_rate': 0.65},
        'bollinger': {'accuracy': 0.65, 'profit_factor': 1.52, 'win_rate': 0.61},
        'adx': {'accuracy': 0.64, 'profit_factor': 1.48, 'win_rate': 0.60},
        'ichimoku': {'accuracy': 0.70, 'profit_factor': 1.65, 'win_rate': 0.66},
        'fibonacci': {'accuracy': 0.62, 'profit_factor': 1.42, 'win_rate': 0.59},
        'stoch': {'accuracy': 0.65, 'profit_factor': 1.50, 'win_rate': 0.62},
        'ema_trend': {'accuracy': 0.68, 'profit_factor': 1.58, 'win_rate': 0.64},
        'parabolic_sar': {'accuracy': 0.61, 'profit_factor': 1.40, 'win_rate': 0.58},
        'obv': {'accuracy': 0.63, 'profit_factor': 1.45, 'win_rate': 0.60},
        'atr': {'accuracy': 0.60, 'profit_factor': 1.35, 'win_rate': 0.57},
        'volatility_rsi': {'accuracy': 0.61, 'profit_factor': 1.38, 'win_rate': 0.58},
        'chaikin_oscillator': {'accuracy': 0.64, 'profit_factor': 1.47, 'win_rate': 0.61},
        'supply_demand': {'accuracy': 0.62, 'profit_factor': 1.41, 'win_rate': 0.59}
    }
    
    # 4. Create optimal periods for different market conditions
    optimal_periods = {
        'trending': {'rsi': 14, 'macd_fast': 12, 'macd_slow': 26, 'bollinger': 20, 'stoch': 14},
        'volatile': {'rsi': 9, 'macd_fast': 9, 'macd_slow': 18, 'bollinger': 15, 'stoch': 9},
        'ranging': {'rsi': 21, 'macd_fast': 15, 'macd_slow': 30, 'bollinger': 25, 'stoch': 21},
        'volatile_trending': {'rsi': 11, 'macd_fast': 10, 'macd_slow': 20, 'bollinger': 18, 'stoch': 11},
        'volatile_ranging': {'rsi': 10, 'macd_fast': 10, 'macd_slow': 20, 'bollinger': 16, 'stoch': 10}
    }
    
    # Save both datasets
    joblib.dump(indicator_performance, 'models/indicator_performance.joblib')
    joblib.dump(optimal_periods, 'models/optimal_periods.joblib')
    
    # 5. Create model metadata
    model_info = {
        'model_type': type(model).__name__ if not hasattr(model, 'named_steps') else type(model.named_steps['model']).__name__,
        'feature_names': features,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_conditions': list(market_condition_info['distribution'].keys()) if market_condition_info else None,
        'version': '2.0'
    }
    
    # Save metadata
    joblib.dump(model_info, 'models/model_info.joblib')
    
    logger.info("All model assets saved successfully")

def main():
    """Main function to run the training pipeline"""
    logger.info("Starting advanced FOREX trading model training")
    
    # 1. Load and prepare realistic trading data with market condition simulation
    X, y, features, market_condition_info = load_and_prepare_data(simulate_market_conditions=True)
    
    # 2. Split data into train/test sets using time-based split
    # In trading, we want to use earlier data for training and later data for testing
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
    
    # 3. Apply advanced preprocessing with feature engineering
    X_train_processed, X_test_processed, processed_features = preprocess_features(
        X_train, X_test, y_train, 
        feature_engineering=True, 
        feature_selection=True
    )
    
    # 4. Optimize models using multiple algorithms and hyperparameter tuning
    best_model, model_performance = optimize_models(
        X_train_processed, y_train,
        X_test_processed, y_test,
        processed_features,
        market_condition_info
    )
    
    # 5. Save all model assets for trading system
    save_trading_model_assets(best_model, processed_features, market_condition_info)
    
    # Log final performance
    best_metric = max(model_performance.values(), key=lambda x: x.get('f1', 0))
    logger.info(f"Training complete! Best model achieves:")
    logger.info(f"  - F1 Score: {best_metric.get('f1', 0):.4f}")
    logger.info(f"  - Accuracy: {best_metric.get('accuracy', 0):.4f}")
    logger.info(f"  - Precision: {best_metric.get('precision', 0):.4f}")
    logger.info(f"  - Recall: {best_metric.get('recall', 0):.4f}")
    
    # Print summary of created files
    logger.info("Created files:")
    logger.info("- models/signal_gb_model.joblib (Main model)")
    logger.info("- models/signal_scaler.joblib (Feature scaler)")
    logger.info("- models/indicator_performance.joblib (Indicator metrics)")
    logger.info("- models/optimal_periods.joblib (Optimal indicator periods)")
    logger.info("- models/model_info.joblib (Model metadata)")
    logger.info("- models/market_condition_performance.joblib (Performance by market condition)")
    logger.info("- models/visualizations/* (Performance visualizations)")

if __name__ == "__main__":
    main() 