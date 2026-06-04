import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    print("Attempting to import AdvancedAnalysis...")
    from src.core.advanced_analysis import AdvancedAnalysis, TF_AVAILABLE
    print(f"Import successful. TF_AVAILABLE: {TF_AVAILABLE}")
    
    if TF_AVAILABLE:
        print("TensorFlow loaded successfully via AdvancedAnalysis.")
    else:
        print("TensorFlow failed to load in AdvancedAnalysis.")
        sys.exit(1)

except Exception as e:
    print(f"Error during import: {e}")
    sys.exit(1)
