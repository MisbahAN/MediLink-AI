#!/usr/bin/env python3
"""
Setup script for NLP dependencies required by the Mistral service.
Run this after installing requirements.txt to set up spaCy language models.
"""

import subprocess
import sys

def run_command(command):
    """Run a command and handle errors."""
    try:
        print(f"Running: {command}")
        subprocess.check_call(command, shell=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {command}: {e}")
        return False

def main():
    """Install required spaCy language models."""
    print("🔧 Setting up NLP dependencies for MediLink-AI...")
    
    # Install basic English model (required)
    success1 = run_command("python -m spacy download en_core_web_sm")
    
    # Try to install medical model (optional, may not be available)
    print("\n📋 Attempting to install medical NLP model (optional)...")
    success2 = run_command("python -m spacy download en_core_sci_md")
    
    if not success2:
        print("⚠️  Medical model not available, will use general English model")
    
    if success1:
        print("\n✅ NLP setup complete! The Mistral service can now perform entity extraction.")
        print("📝 Note: Medical model provides better accuracy for medical documents.")
    else:
        print("\n❌ Failed to install required spaCy models.")
        print("🔍 Please run manually: python -m spacy download en_core_web_sm")
        sys.exit(1)

if __name__ == "__main__":
    main()