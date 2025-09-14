#!/usr/bin/env python3
"""
Smart Bus System Setup Script

This script automates the setup process for the Smart Bus System.
It handles environment setup, dependency installation, and initial data generation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    if not os.path.exists('venv'):
        return run_command('python -m venv venv', 'Creating virtual environment')
    else:
        print("✅ Virtual environment already exists")
        return True

def get_activation_command():
    """Get the correct activation command based on OS"""
    if platform.system() == 'Windows':
        return 'venv\\Scripts\\activate'
    else:
        return 'source venv/bin/activate'

def install_dependencies():
    """Install Python dependencies"""
    if platform.system() == 'Windows':
        pip_command = 'venv\\Scripts\\pip'
    else:
        pip_command = 'venv/bin/pip'
    
    return run_command(f'{pip_command} install -r requirements.txt', 'Installing dependencies')

def create_directories():
    """Create necessary directories"""
    directories = ['data/raw', 'data/clean', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Created necessary directories")
    return True

def generate_synthetic_data():
    """Generate synthetic data"""
    if platform.system() == 'Windows':
        python_command = 'venv\\Scripts\\python'
    else:
        python_command = 'venv/bin/python'
    
    return run_command(f'{python_command} data/generate_synthetic.py', 'Generating synthetic data')

def create_env_file():
    """Create .env file with default settings"""
    env_content = """# Smart Bus System Environment Variables
FLASK_ENV=development
DATABASE_URL=sqlite:///smart_bus.db
SECRET_KEY=dev-secret-key-change-in-production
PORT=5000
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default settings")
    else:
        print("✅ .env file already exists")
    return True

def main():
    """Main setup function"""
    print("🚌 Smart Bus System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Generate synthetic data
    if not generate_synthetic_data():
        print("⚠️  Synthetic data generation failed, but you can run it manually later")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print(f"1. Activate virtual environment: {get_activation_command()}")
    print("2. Run the application: python run.py")
    print("3. Open your browser: http://localhost:5000")
    print("\nFor production deployment, use: pip install -r requirements-prod.txt")

if __name__ == '__main__':
    main()
