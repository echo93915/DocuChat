"""Entry point to run the DocuChat Streamlit application."""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the Streamlit app
from src.app_streamlit import main

if __name__ == "__main__":
    main()
