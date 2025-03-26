import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Added {project_root} to Python path")
print("Setup complete - you can now import modules from the src directory") 