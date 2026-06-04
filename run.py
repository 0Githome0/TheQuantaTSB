import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gui.app import main

if __name__ == "__main__":
    main()
