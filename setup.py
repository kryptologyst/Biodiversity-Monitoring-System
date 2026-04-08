#!/usr/bin/env python3
"""Setup script for biodiversity monitoring system."""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Biodiversity Monitoring System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/raw", "data/processed", "data/external",
        "assets", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run basic tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but setup continues...")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python scripts/train.py")
    print("2. Launch demo: streamlit run demo/app.py")
    print("3. Explore notebook: jupyter notebook notebooks/exploration.ipynb")

if __name__ == "__main__":
    main()
