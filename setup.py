"""
Setup script for Financial Sentiment Analysis
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "transformers", "pandas", "numpy", 
            "scikit-learn", "seaborn", "matplotlib", 
            "tensorflow", "requests"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def create_data_directory():
    """Create data directory if it doesn't exist"""
    os.makedirs("data", exist_ok=True)
    print("✅ Data directory created/verified")


def main():
    """Setup the project"""
    print("="*50)
    print("FINANCIAL SENTIMENT ANALYSIS - SETUP")
    print("="*50)
    
    # Install packages
    if not install_requirements():
        print("❌ Setup failed!")
        sys.exit(1)
    
    # Create directories
    create_data_directory()
    
    print("\n" + "="*50)
    print("✅ Setup complete!")
    print("Run: python main.py to start training")
    print("="*50)


if __name__ == "__main__":
    main()
