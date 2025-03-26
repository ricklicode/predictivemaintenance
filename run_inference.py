#!/usr/bin/env python
"""
Command-line script to make predictions with a trained predictive maintenance model.
Example usage:
    python run_inference.py --model_path saved_models/final_model.h5 --data_path new_data.csv
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the inference function
from src.inference import parse_args, main

if __name__ == "__main__":
    # Run the inference
    main() 