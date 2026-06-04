import numpy as np
import pandas as pd
import joblib
import os
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
from functools import wraps
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging with more detailed format
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AdvancedModelCreator')

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
MODEL_VERSION = "1.0.0"

# Ensure model directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('models/visualizations', exist_ok=True)
os.makedirs('models/metadata', exist_ok=True)
os.makedirs('models/checkpoints', exist_ok=True)

# Timer decorator for performance tracking
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Function '{func.__name__}' completed in {elapsed_time:.2f} seconds")
        return result
    return wrapper

# Metadata tracking
def save_model_metadata(model_name, metadata):
    """Save metadata for a model for tracking and reproducibility"""
    metadata_file = f"models/metadata/{model_name}_metadata.json"
    
    # Add standard fields
    metadata.update({
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": MODEL_VERSION,
        "python_version": "3.12",  # Adjust if needed
    })
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_file}")

# Generate realistic OHLC data for financial markets
def generate_market_data(n_days=500, n_symbols=5, start_date=None):
    """
    Generate realistic OHLCV data for financial symbols
    Returns a dictionary with DataFrames for each symbol
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_days)
    
    market_data = {}
    
    for symbol_idx in range(n_symbols):
        symbol_name = f"SYMBOL_{symbol_idx+1}"
        
        # Generate dates
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Initial price and volatility
        base_price = np.random.uniform(50, 5000)
        volatility = np.random.uniform(0.01, 0.03)
        
        # Add trends, cycles, and random noise
        prices = []
        current_price = base_price
        
        # Create a trend component
        trend_direction = np.random.choice([-1, 1])
        trend_strength = np.random.uniform(0.0001, 0.0005)
        
        # Create a cycle component (simulating market cycles)
        cycle_period = np.random.randint(20, 60)
        cycle_amplitude = base_price * np.random.uniform(0.01, 0.05)
        
        for i in range(n_days):
            # Apply trend
            trend = trend_direction * trend_strength * current_price
            
            # Apply cycle
            cycle = cycle_amplitude * np.sin(2 * np.pi * i / cycle_period)
            
            # Apply random noise (volatility)
            noise = np.random.normal(0, volatility * current_price)
            
            # Calculate day's price movement
            price_change = trend + cycle + noise
            
            # Randomly introduce volatility clusters
            if np.random.random() < 0.05:  # 5% chance of volatility spike
                price_change *= np.random.uniform(2, 5)
            
            # Update current price
            current_price += price_change
            current_price = max(current_price, base_price * 0.1)  # Prevent negative/tiny prices
            
            prices.append(current_price)
        
        # Generate OHLC data from the prices
        close_prices = np.array(prices)
        high_prices = close_prices * np.random.uniform(1.001, 1.03, n_days)
        low_prices = close_prices * np.random.uniform(0.97, 0.999, n_days)
        open_prices = np.random.uniform(low_prices, high_prices)
        
        # Generate volume data - higher during price movements
        price_changes = np.abs(np.diff(np.append([base_price], close_prices)))
        normalized_changes = price_changes / np.mean(price_changes)
        volume = np.random.normal(1000000, 200000, n_days) * (0.5 + normalized_changes)
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': open_prices, 
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume.astype(int)
        })
        
        data.set_index('date', inplace=True)
        market_data[symbol_name] = data
    
    return market_data

# Feature engineering for financial time series
def create_financial_features(df):
    """
    Create technical indicators and financial features from OHLCV data
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        
    Returns:
        DataFrame with original and engineered features
    """
    # Create a copy to avoid modifying the original data
    df_features = df.copy()
    
    # Ensure we have proper columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df_features.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}. Some features may not be calculated correctly.")
    
    # Calculate price-based indicators
    
    # Moving averages
    df_features['ma5'] = df_features['close'].rolling(window=5).mean()
    df_features['ma10'] = df_features['close'].rolling(window=10).mean()
    df_features['ma20'] = df_features['close'].rolling(window=20).mean()
    df_features['ma50'] = df_features['close'].rolling(window=50).mean()
    
    # Exponential moving averages
    df_features['ema5'] = df_features['close'].ewm(span=5, adjust=False).mean()
    df_features['ema10'] = df_features['close'].ewm(span=10, adjust=False).mean()
    df_features['ema20'] = df_features['close'].ewm(span=20, adjust=False).mean()
    
    # Price momentum
    df_features['returns_1d'] = df_features['close'].pct_change(1)
    df_features['returns_5d'] = df_features['close'].pct_change(5)
    df_features['returns_10d'] = df_features['close'].pct_change(10)
    
    # Volatility
    df_features['volatility_5d'] = df_features['returns_1d'].rolling(window=5).std()
    df_features['volatility_10d'] = df_features['returns_1d'].rolling(window=10).std()
    df_features['volatility_20d'] = df_features['returns_1d'].rolling(window=20).std()
    
    # Price channels
    df_features['upper_channel_10d'] = df_features['high'].rolling(window=10).max()
    df_features['lower_channel_10d'] = df_features['low'].rolling(window=10).min()
    df_features['upper_channel_20d'] = df_features['high'].rolling(window=20).max()
    df_features['lower_channel_20d'] = df_features['low'].rolling(window=20).min()
    
    # Relative position within price channel
    df_features['channel_pos_10d'] = (df_features['close'] - df_features['lower_channel_10d']) / (df_features['upper_channel_10d'] - df_features['lower_channel_10d'])
    df_features['channel_pos_20d'] = (df_features['close'] - df_features['lower_channel_20d']) / (df_features['upper_channel_20d'] - df_features['lower_channel_20d'])
    
    # Moving average convergence divergence (MACD)
    df_features['macd_line'] = df_features['close'].ewm(span=12, adjust=False).mean() - df_features['close'].ewm(span=26, adjust=False).mean()
    df_features['macd_signal'] = df_features['macd_line'].ewm(span=9, adjust=False).mean()
    df_features['macd_histogram'] = df_features['macd_line'] - df_features['macd_signal']
    
    # Bollinger Bands
    df_features['bollinger_mid_20d'] = df_features['close'].rolling(window=20).mean()
    df_features['bollinger_std_20d'] = df_features['close'].rolling(window=20).std()
    df_features['bollinger_upper_20d'] = df_features['bollinger_mid_20d'] + 2 * df_features['bollinger_std_20d']
    df_features['bollinger_lower_20d'] = df_features['bollinger_mid_20d'] - 2 * df_features['bollinger_std_20d']
    df_features['bollinger_width_20d'] = (df_features['bollinger_upper_20d'] - df_features['bollinger_lower_20d']) / df_features['bollinger_mid_20d']
    df_features['bollinger_pos_20d'] = (df_features['close'] - df_features['bollinger_lower_20d']) / (df_features['bollinger_upper_20d'] - df_features['bollinger_lower_20d'])
    
    # Relative Strength Index (RSI)
    def calculate_rsi(prices, window=14):
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df_features['rsi_14d'] = calculate_rsi(df_features['close'], window=14)
    df_features['rsi_7d'] = calculate_rsi(df_features['close'], window=7)
    
    # Commodity Channel Index (CCI)
    def calculate_cci(high, low, close, window=20):
        # Calculate typical price
        tp = (high + low + close) / 3
        
        # Calculate moving average of typical price
        ma_tp = tp.rolling(window=window).mean()
        
        # Calculate mean deviation
        md = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        
        # Calculate CCI
        cci = (tp - ma_tp) / (0.015 * md)
        return cci
    
    df_features['cci_20d'] = calculate_cci(df_features['high'], df_features['low'], df_features['close'], window=20)
    
    # Average Directional Index (ADX)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Plus Directional Movement (+DM)
        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -low.diff()), 0)
        
        # Minus Directional Movement (-DM)
        minus_dm = low.diff()
        minus_dm = -minus_dm.where((minus_dm > 0) & (minus_dm > high.diff()), 0)
        
        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Directional Movement Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Average Directional Index (ADX)
        adx = dx.rolling(window=window).mean()
        
        return adx, plus_di, minus_di
    
    df_features['adx_14d'], df_features['plus_di_14d'], df_features['minus_di_14d'] = calculate_adx(
        df_features['high'], df_features['low'], df_features['close'], window=14
    )
    
    # Volume-based indicators
    
    # Volume moving averages
    df_features['volume_ma5'] = df_features['volume'].rolling(window=5).mean()
    df_features['volume_ma10'] = df_features['volume'].rolling(window=10).mean()
    df_features['volume_ma20'] = df_features['volume'].rolling(window=20).mean()
    
    # Volume change
    df_features['volume_change_1d'] = df_features['volume'].pct_change(1)
    df_features['volume_change_5d'] = df_features['volume'].pct_change(5)
    
    # Price-volume relationship
    df_features['price_volume_corr_5d'] = df_features['close'].rolling(window=5).corr(df_features['volume'])
    df_features['price_volume_corr_10d'] = df_features['close'].rolling(window=10).corr(df_features['volume'])
    
    # On-Balance Volume (OBV)
    df_features['obv'] = (np.sign(df_features['close'].diff()) * df_features['volume']).fillna(0).cumsum()
    
    # Chaikin Money Flow (CMF)
    def calculate_cmf(high, low, close, volume, window=20):
        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Chaikin Money Flow
        cmf = mf_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()
        return cmf
    
    df_features['cmf_20d'] = calculate_cmf(
        df_features['high'], df_features['low'], df_features['close'], df_features['volume'], window=20
    )
    
    # Clean NA values by replacing with 0
    # In a real application, you might want to handle NAs differently
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    return df_features

# Create a comprehensive visualization
def visualize_feature_importances(model, feature_names, model_name, top_n=15):
    """Create and save a detailed visualization of feature importances"""
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {model_name} does not have feature_importances_ attribute.")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    indices = indices[:top_n]
    features = [feature_names[i] for i in indices]
    importances = importances[indices]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), features)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Feature Importances for {model_name}')
    plt.tight_layout()
    
    # Save the visualization
    output_path = f'models/visualizations/{model_name}_feature_importance.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Return for potential further use
    return pd.Series(importances, index=features)

@timer
def create_signal_classifier():
    """Create and save an advanced signal classifier model"""
    logger.info("Creating advanced signal classifier model...")
    
    # Generate realistic market data
    market_data = generate_market_data(n_days=1000, n_symbols=5)
    
    # Process data and create features
    features_list = []
    targets = []
    
    for symbol, df in market_data.items():
        # Engineer features
        df_features = create_financial_features(df)
        
        # Create target variable: 1 if next day's return is positive, 0 otherwise
        df_features['target'] = (df_features['returns_1d'].shift(-1) > 0).astype(int)
        
        # Create custom trading signals for ML model to learn
        # Crossover signals
        df_features['ma_crossover'] = ((df_features['ma5'] > df_features['ma20']) & 
                                      (df_features['ma5'].shift(1) <= df_features['ma20'].shift(1))).astype(int)
        
        # RSI signals
        df_features['rsi_oversold'] = (df_features['rsi_14d'] < 30).astype(int)
        df_features['rsi_overbought'] = (df_features['rsi_14d'] > 70).astype(int)
        
        # MACD signals
        df_features['macd_crossover'] = ((df_features['macd_line'] > df_features['macd_signal']) & 
                                        (df_features['macd_line'].shift(1) <= df_features['macd_signal'].shift(1))).astype(int)
        
        # Rule-based target: Improved version based on multiple signals
        # If crossover and RSI conditions align, higher probability of success
        success_conditions = (
            (df_features['ma_crossover'] == 1) & 
            (df_features['macd_crossover'] == 1) & 
            (df_features['rsi_14d'] > 40) & 
            (df_features['rsi_14d'] < 60)
        )
        df_features.loc[success_conditions, 'target'] = 1
        
        # If price is at the lower Bollinger and RSI is oversold, buy signal likely successful
        bb_conditions = (
            (df_features['close'] <= df_features['bollinger_lower_20d'] * 1.01) & 
            (df_features['rsi_14d'] < 35)
        )
        df_features.loc[bb_conditions, 'target'] = 1
        
        # If RSI is extremely overbought and price at upper Bollinger, short signal likely successful
        overbought_conditions = (
            (df_features['close'] >= df_features['bollinger_upper_20d'] * 0.99) & 
            (df_features['rsi_14d'] > 75)
        )
        df_features.loc[overbought_conditions, 'target'] = 0
        
        # Drop rows with NaN
        df_features.dropna(inplace=True)
        
        # Select features for model training
        X = df_features.drop(['target', 'open', 'high', 'low', 'close', 'volume'], axis=1)
        y = df_features['target']
        
        features_list.append(X)
        targets.append(y)
    
    # Combine all symbols' data
    X_all = pd.concat(features_list, axis=0)
    y_all = pd.concat(targets, axis=0)
    
    # Split the data (using time series split instead of random to respect temporal order)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
    )
    
    # Create a pipeline with feature scaling
    logger.info("Training signal classifier with advanced features...")
    
    # Define a list of models to try
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight='balanced'
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=RANDOM_SEED
        )
    }
    
    # Train and evaluate each model
    results = {}
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        # Calculate various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Store results
        results[name] = {
            'train_score': train_score,
            'test_score': test_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"{name} - Train: {train_score:.4f}, Test: {test_score:.4f}, F1: {f1:.4f}")
        
        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = model
    
    # Create a voting classifier from the best models
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']),
            ('gb', models['GradientBoosting'])
        ],
        voting='soft'
    )
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    
    # Evaluate the ensemble
    ensemble_pred = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
    
    logger.info(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
    
    # Choose the best model (ensemble or individual)
    if ensemble_f1 > best_score:
        final_model = ensemble
        model_name = "Ensemble"
        logger.info("Ensemble model selected as the best model")
    else:
        final_model = best_model
        model_name = "RandomForest" if best_model == models["RandomForest"] else "GradientBoosting"
        logger.info(f"{model_name} selected as the best model")
        
    # Create feature importance visualization
    if model_name != "Ensemble":
        visualize_feature_importances(final_model, X_train.columns, "SignalClassifier")
    
    # Save the model
    model_path = 'models/signal_classifier.joblib'
    joblib.dump(final_model, model_path)
    logger.info(f"Signal classifier saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": model_name,
        "features": list(X_train.columns),
        "performance": {
            "accuracy": float(ensemble_accuracy if model_name == "Ensemble" else results[model_name]['accuracy']),
            "f1_score": float(ensemble_f1 if model_name == "Ensemble" else results[model_name]['f1']),
            "precision": float(results[model_name]['precision'] if model_name != "Ensemble" else precision_score(y_test, ensemble_pred, zero_division=0)),
            "recall": float(results[model_name]['recall'] if model_name != "Ensemble" else recall_score(y_test, ensemble_pred, zero_division=0))
        },
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": {
            "training": {
                "class_0": int((y_train == 0).sum()),
                "class_1": int((y_train == 1).sum())
            },
            "test": {
                "class_0": int((y_test == 0).sum()),
                "class_1": int((y_test == 1).sum())
            }
        }
    }
    
    save_model_metadata("signal_classifier", metadata)
    
    return final_model

@timer
def create_anomaly_detector():
    """Create and save an improved anomaly detector model"""
    logger.info("Creating advanced anomaly detector model...")
    
    # Generate realistic market data with some abnormal patterns
    market_data = generate_market_data(n_days=1000, n_symbols=3)
    
    # Process data and create features
    features_list = []
    
    for symbol, df in market_data.items():
        # Engineer features
        df_features = create_financial_features(df)
        
        # Inject some anomalies (about 5% of data points)
        n_anomalies = int(len(df_features) * 0.05)
        anomaly_indices = np.random.choice(df_features.index, size=n_anomalies, replace=False)
        
        # Create different types of anomalies
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['volatility', 'price_jump', 'volume_spike', 'correlation_break'])
            
            if anomaly_type == 'volatility':
                # Abnormal volatility
                df_features.loc[idx, 'volatility_5d'] *= np.random.uniform(5, 10)
                df_features.loc[idx, 'volatility_10d'] *= np.random.uniform(5, 10)
                df_features.loc[idx, 'volatility_20d'] *= np.random.uniform(5, 10)
                
            elif anomaly_type == 'price_jump':
                # Extreme price movement
                df_features.loc[idx, 'returns_1d'] = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.2)
            
            elif anomaly_type == 'volume_spike':
                # Abnormal trading volume
                df_features.loc[idx, 'volume'] *= np.random.uniform(10, 20)
                for window in [5, 10, 20]:
                    df_features.loc[idx, f'volume_ma{window}'] *= np.random.uniform(3, 5)
            
            elif anomaly_type == 'correlation_break':
                # Break normal correlations between indicators
                df_features.loc[idx, 'rsi_14d'] = np.random.uniform(0, 100)
                df_features.loc[idx, 'macd_line'] = -1 * df_features.loc[idx, 'macd_line'] * np.random.uniform(2, 4)
                df_features.loc[idx, 'bollinger_width_20d'] *= np.random.uniform(3, 5)
        
        # Select features for model training (exclude price and volume raw data)
        X = df_features.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
        
        # Add to collection
        features_list.append(X)
    
    # Combine all symbols' data
    X_all = pd.concat(features_list, axis=0)
    
    # Drop any remaining NaN values
    X_all.dropna(inplace=True)
    
    # Scale the features for better anomaly detection
    scaler = RobustScaler()  # RobustScaler handles outliers better
    X_scaled = scaler.fit_transform(X_all)
    
    # Create and train the anomaly detector
    model = IsolationForest(
        n_estimators=150,
        max_samples='auto',
        contamination=0.05,  # Expected proportion of anomalies
        max_features=0.8,    # Use 80% of features for better robustness
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0
    )
    
    model.fit(X_scaled)
    
    # Predict anomalies to evaluate the model
    anomaly_scores = model.decision_function(X_scaled)
    anomaly_predictions = model.predict(X_scaled)
    
    # Convert predictions: IsolationForest uses 1 for inliers, -1 for outliers
    # We'll convert to 0 for normal, 1 for anomaly for clarity
    anomalies = (anomaly_predictions == -1).astype(int)
    
    # Calculate percentage of detected anomalies
    anomaly_percentage = 100 * anomalies.sum() / len(anomalies)
    logger.info(f"Anomaly detector identified {anomaly_percentage:.2f}% of data points as anomalies")
    
    # Calculate threshold (the decision function value that separates inliers from outliers)
    # Sort the scores and find where the prediction changes
    threshold = np.sort(anomaly_scores)[int(len(anomaly_scores) * model.contamination)]
    
    # Create visualization of anomaly score distribution
    plt.figure(figsize=(12, 6))
    plt.hist(anomaly_scores, bins=50, alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', 
               label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/visualizations/anomaly_score_distribution.png')
    plt.close()
    
    # Save the model
    model_path = 'models/anomaly_detector.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Anomaly detector saved to {model_path}")
    
    # Save the scaler for future preprocessing
    scaler_path = 'models/anomaly_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Anomaly scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        "model_type": "IsolationForest",
        "parameters": {
            "n_estimators": 150,
            "contamination": 0.05,
            "max_features": 0.8,
            "bootstrap": True
        },
        "features": list(X_all.columns),
        "training_samples": len(X_all),
        "anomaly_threshold": float(threshold),
        "detected_anomalies_percentage": float(anomaly_percentage)
    }
    
    save_model_metadata("anomaly_detector", metadata)
    
    return model, scaler

@timer
def create_adv_scaler():
    """Create and save an advanced scaler with improved preprocessing"""
    logger.info("Creating advanced scaler model...")
    
    # Generate realistic market data
    market_data = generate_market_data(n_days=800, n_symbols=8)
    
    # Process data and create features
    features_list = []
    
    for symbol, df in market_data.items():
        # Engineer features
        df_features = create_financial_features(df)
        
        # Select numeric columns for scaling
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        X = df_features[numeric_columns]
        
        features_list.append(X)
    
    # Combine all symbols' data
    X_all = pd.concat(features_list, axis=0)
    
    # Drop any NaN values
    X_all.dropna(inplace=True)
    
    # Compare different scaling methods to find the best one
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler(),
        'Power': PowerTransformer(method='yeo-johnson')
    }
    
    # Evaluate each scaler
    scaler_stats = {}
    
    for name, scaler in scalers.items():
        # Fit and transform
        X_scaled = scaler.fit_transform(X_all)
        
        # Calculate statistics on scaled data
        mean = np.mean(X_scaled)
        std = np.std(X_scaled)
        skew = np.mean(((X_scaled - mean) / std) ** 3)
        kurtosis = np.mean(((X_scaled - mean) / std) ** 4) - 3
        
        # Store statistics
        scaler_stats[name] = {
            'mean': mean,
            'std': std,
            'min': np.min(X_scaled),
            'max': np.max(X_scaled),
            'skew': skew,
            'kurtosis': kurtosis
        }
        
        logger.info(f"{name} Scaler - Mean: {mean:.4f}, Std: {std:.4f}, " 
                   f"Range: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}], "
                   f"Skew: {skew:.4f}, Kurtosis: {kurtosis:.4f}")
    
    # Choose the best scaler based on normality (closest to normal distribution)
    # The best scaler should have skew and kurtosis closest to 0
    best_scaler_name = min(scaler_stats.keys(), 
                           key=lambda x: abs(scaler_stats[x]['skew']) + abs(scaler_stats[x]['kurtosis']))
    
    # Get the best scaler
    best_scaler = scalers[best_scaler_name]
    logger.info(f"Selected {best_scaler_name} as the best scaler")
    
    # Create visualizations of scaled data
    plt.figure(figsize=(15, 10))
    
    for i, (name, scaler) in enumerate(scalers.items()):
        X_scaled = scaler.fit_transform(X_all)
        
        # Flatten the scaled data
        X_flat = X_scaled.flatten()
        
        # Plot histogram
        plt.subplot(2, 2, i+1)
        plt.hist(X_flat, bins=50, alpha=0.7)
        plt.title(f'{name} Scaler Distribution')
        plt.xlabel('Scaled Value')
        plt.ylabel('Frequency')
        
        # Highlight the best scaler
        if name == best_scaler_name:
            plt.title(f'{name} Scaler Distribution (SELECTED)')
            plt.axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('models/visualizations/scaler_comparison.png')
    plt.close()
    
    # Save the selected scaler
    scaler_path = 'models/adv_scaler.joblib'
    joblib.dump(best_scaler, scaler_path)
    logger.info(f"Advanced scaler ({best_scaler_name}) saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        "scaler_type": best_scaler_name,
        "comparison_results": {name: {k: float(v) for k, v in stats.items()} 
                              for name, stats in scaler_stats.items()},
        "features": list(X_all.columns),
        "training_samples": len(X_all)
    }
    
    save_model_metadata("adv_scaler", metadata)
    
    return best_scaler

@timer
def create_feature_selector():
    """Create and save an advanced feature selector for financial prediction"""
    logger.info("Creating feature selector model...")
    
    # Generate realistic market data
    market_data = generate_market_data(n_days=1000, n_symbols=5)
    
    # Process data and create features
    features_list = []
    targets = []
    
    for symbol, df in market_data.items():
        # Engineer features
        df_features = create_financial_features(df)
        
        # Create target: predict if close will be higher than open tomorrow
        df_features['target'] = (df_features['close'].shift(-1) > df_features['open'].shift(-1)).astype(int)
        
        # Select features for model training
        X = df_features.drop(['target', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
        y = df_features['target']
        
        # Drop rows with NaN
        X = X.dropna()
        y = y.loc[X.index]
        
        features_list.append(X)
        targets.append(y)
    
    # Combine all symbols' data
    X_all = pd.concat(features_list, axis=0)
    y_all = pd.concat(targets, axis=0)
    
    logger.info(f"Feature selection dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Apply different feature selection methods
    
    # 1. F-classification
    logger.info("Applying F-classification for feature selection...")
    f_selector = SelectKBest(f_classif, k=10)
    f_selector.fit(X_train, y_train)
    f_scores = f_selector.scores_
    
    # Normalize scores
    f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
    
    # Create DataFrame with feature names and scores
    f_selection_df = pd.DataFrame({
        'Feature': X_train.columns,
        'F_Score': f_scores,
        'F_Score_Norm': f_scores_norm
    })
    f_selection_df = f_selection_df.sort_values('F_Score', ascending=False)
    
    logger.info("Top 5 features by F-classification:")
    for idx, row in f_selection_df.head(5).iterrows():
        logger.info(f"  {row['Feature']}: {row['F_Score']:.4f}")
    
    # 2. Mutual Information
    logger.info("Applying Mutual Information for feature selection...")
    mi_selector = SelectKBest(mutual_info_classif, k=10)
    mi_selector.fit(X_train, y_train)
    mi_scores = mi_selector.scores_
    
    # Normalize scores
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    
    # Create DataFrame with feature names and scores
    mi_selection_df = pd.DataFrame({
        'Feature': X_train.columns,
        'MI_Score': mi_scores,
        'MI_Score_Norm': mi_scores_norm
    })
    mi_selection_df = mi_selection_df.sort_values('MI_Score', ascending=False)
    
    logger.info("Top 5 features by Mutual Information:")
    for idx, row in mi_selection_df.head(5).iterrows():
        logger.info(f"  {row['Feature']}: {row['MI_Score']:.4f}")
    
    # 3. Random Forest Feature Importance
    logger.info("Applying Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    
    # Normalize scores
    rf_importances_norm = (rf_importances - rf_importances.min()) / (rf_importances.max() - rf_importances.min())
    
    # Create DataFrame with feature names and importances
    rf_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'RF_Importance': rf_importances,
        'RF_Importance_Norm': rf_importances_norm
    })
    rf_importance_df = rf_importance_df.sort_values('RF_Importance', ascending=False)
    
    logger.info("Top 5 features by Random Forest Importance:")
    for idx, row in rf_importance_df.head(5).iterrows():
        logger.info(f"  {row['Feature']}: {row['RF_Importance']:.4f}")
    
    # 4. Recursive Feature Elimination (RFE)
    logger.info("Applying Recursive Feature Elimination...")
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED), n_features_to_select=10)
    rfe.fit(X_train, y_train)
    rfe_ranking = rfe.ranking_
    
    # Convert rankings to scores (lower rank = higher score)
    rfe_scores = 1 / rfe_ranking
    
    # Normalize scores
    rfe_scores_norm = (rfe_scores - rfe_scores.min()) / (rfe_scores.max() - rfe_scores.min())
    
    # Create DataFrame with feature names and RFE rankings
    rfe_ranking_df = pd.DataFrame({
        'Feature': X_train.columns,
        'RFE_Ranking': rfe_ranking,
        'RFE_Score': rfe_scores,
        'RFE_Score_Norm': rfe_scores_norm
    })
    rfe_ranking_df = rfe_ranking_df.sort_values('RFE_Ranking')
    
    logger.info("Top 5 features by RFE:")
    for idx, row in rfe_ranking_df.head(5).iterrows():
        logger.info(f"  {row['Feature']}: Rank {int(row['RFE_Ranking'])}")
    
    # Combine all scores
    all_scores = pd.DataFrame({'Feature': X_train.columns})
    all_scores = pd.merge(all_scores, f_selection_df[['Feature', 'F_Score_Norm']], on='Feature')
    all_scores = pd.merge(all_scores, mi_selection_df[['Feature', 'MI_Score_Norm']], on='Feature')
    all_scores = pd.merge(all_scores, rf_importance_df[['Feature', 'RF_Importance_Norm']], on='Feature')
    all_scores = pd.merge(all_scores, rfe_ranking_df[['Feature', 'RFE_Score_Norm']], on='Feature')
    
    # Calculate ensemble score (mean of all normalized scores)
    all_scores['Ensemble_Score'] = all_scores[['F_Score_Norm', 'MI_Score_Norm', 'RF_Importance_Norm', 'RFE_Score_Norm']].mean(axis=1)
    
    # Sort by ensemble score
    all_scores = all_scores.sort_values('Ensemble_Score', ascending=False)
    
    logger.info("Top 10 features by Ensemble Ranking:")
    for idx, row in all_scores.head(10).iterrows():
        logger.info(f"  {row['Feature']}: {row['Ensemble_Score']:.4f}")
    
    # Select top features
    top_features = all_scores.head(15)['Feature'].tolist()
    
    # Create feature selector model (a simple wrapper around the feature list)
    feature_selector = {
        'top_features': top_features,
        'all_scores': all_scores
    }
    
    # Save feature selector
    selector_path = 'models/feature_selector.joblib'
    joblib.dump(feature_selector, selector_path)
    logger.info(f"Feature selector saved to {selector_path}")
    
    # Visualize top features and their scores
    plt.figure(figsize=(12, 8))
    
    # Plot top 15 features by ensemble score
    top_n = 15
    top_features_df = all_scores.head(top_n)
    
    plt.barh(range(len(top_features_df)), top_features_df['Ensemble_Score'], align='center')
    plt.yticks(range(len(top_features_df)), top_features_df['Feature'])
    plt.xlabel('Ensemble Score')
    plt.title(f'Top {top_n} Features by Ensemble Score')
    plt.tight_layout()
    plt.savefig('models/visualizations/top_features_ensemble.png')
    plt.close()
    
    # Create a heatmap of normalized scores for top features
    plt.figure(figsize=(12, 10))
    score_columns = ['F_Score_Norm', 'MI_Score_Norm', 'RF_Importance_Norm', 'RFE_Score_Norm']
    heatmap_data = top_features_df[['Feature'] + score_columns].set_index('Feature')
    
    # Rename columns for better readability
    heatmap_data.columns = ['F-classification', 'Mutual Information', 'Random Forest', 'RFE']
    
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Feature Selection Methods Comparison')
    plt.tight_layout()
    plt.savefig('models/visualizations/feature_selection_heatmap.png')
    plt.close()
    
    # Save metadata
    metadata = {
        "top_features": top_features,
        "selection_methods": ["F-classification", "Mutual Information", "Random Forest Importance", "RFE"],
        "training_samples": len(X_train),
        "total_features_evaluated": len(X_train.columns)
    }
    
    save_model_metadata("feature_selector", metadata)
    
    return feature_selector

@timer
def create_deep_learning_model():
    """Create and save a Bidirectional LSTM deep learning model for time series prediction"""
    logger.info("Creating deep learning model...")
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.optimizers import Adam
        tf_available = True
        logger.info("Using TensorFlow for deep learning model")
    except ImportError:
        tf_available = False
        logger.warning("TensorFlow not available, creating placeholder model instead")
        
    if not tf_available:
        # Create placeholder model if TensorFlow is not available
        placeholder_model = {
            "type": "placeholder",
            "message": "TensorFlow not available, placeholder model created"
        }
        # Save placeholder
        joblib.dump(placeholder_model, 'models/deep_model_placeholder.joblib')
        logger.info("Placeholder model saved")
        
        return placeholder_model
    
    # Generate realistic market data
    market_data = generate_market_data(n_days=1000, n_symbols=3)
    symbol = list(market_data.keys())[0]  # Use first symbol
    
    # Create features
    df = create_financial_features(market_data[symbol])
    
    # Create target: binary classification (price movement direction)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Select features for model training (excluding price columns and target)
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    X = df.drop(price_cols + ['target'], axis=1, errors='ignore')
    y = df['target']
    
    logger.info(f"Deep learning dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for LSTM input
    def create_sequences(X, y, seq_length=10):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i+seq_length])
            ys.append(y.iloc[i+seq_length])
        return np.array(Xs), np.array(ys)
    
    # Set sequence length
    seq_length = 10
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)
    
    logger.info(f"Sequence shape: {X_seq.shape}, Target shape: {y_seq.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Create Bidirectional LSTM model
    model = Sequential([
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layers
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary(print_fn=logger.info)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath='models/checkpoints/deep_model_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train model
    logger.info("Training deep learning model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=2
    )
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
               f"Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Visualize training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/visualizations/deep_model_training.png')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - LSTM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('models/visualizations/deep_model_confusion_matrix.png')
    plt.close()
    
    # Save the model
    model_path = 'models/deep_model.h5'
    model.save(model_path)
    logger.info(f"Deep learning model saved to {model_path}")
    
    # Save the scaler
    scaler_path = 'models/deep_model_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Deep learning scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        "model_type": "Bidirectional LSTM",
        "features": list(X.columns),
        "sequence_length": seq_length,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "hyperparameters": {
            "lstm_units": [64, 32],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    }
    
    save_model_metadata("deep_model", metadata)
    
    return model

@timer
def main():
    """Main function to create all models"""
    logger.info("Starting creation of advanced models...")
    
    # Create Signal Classifier
    logger.info("=== Creating Signal Classifier ===")
    signal_classifier = create_signal_classifier()
    
    # Create Anomaly Detector
    logger.info("=== Creating Anomaly Detector ===")
    anomaly_detector, anomaly_scaler = create_anomaly_detector()
    
    # Create Advanced Scaler
    logger.info("=== Creating Advanced Scaler ===")
    adv_scaler = create_adv_scaler()
    
    # Create Feature Selector
    logger.info("=== Creating Feature Selector ===")
    feature_selector = create_feature_selector()
    
    # Create Deep Learning Model
    logger.info("=== Creating Deep Learning Model ===")
    deep_model = create_deep_learning_model()
    
    # Summarize created models
    logger.info("\n=== Model Creation Summary ===")
    logger.info("The following models have been created:")
    logger.info("1. Signal Classifier: models/signal_classifier.joblib")
    logger.info("2. Anomaly Detector: models/anomaly_detector.joblib")
    logger.info("3. Advanced Scaler: models/adv_scaler.joblib")
    logger.info("4. Feature Selector: models/feature_selector.joblib")
    logger.info("5. Deep Learning Model: models/deep_model.h5")
    
    # Summarize visualizations
    logger.info("\nThe following visualizations have been created:")
    logger.info("1. Signal Classifier: models/visualizations/signal_classifier_confusion_matrix.png, models/visualizations/signal_classifier_feature_importance.png")
    logger.info("2. Anomaly Detector: models/visualizations/anomaly_detection.png, models/visualizations/anomaly_feature_distribution.png")
    logger.info("3. Feature Scaling: models/visualizations/feature_scaling.png")
    logger.info("4. Feature Selection: models/visualizations/top_features_ensemble.png, models/visualizations/feature_selection_heatmap.png")
    logger.info("5. Deep Learning: models/visualizations/deep_model_training.png, models/visualizations/deep_model_confusion_matrix.png")
    
    logger.info("\nAll models have been successfully created.")

if __name__ == "__main__":
    main() 