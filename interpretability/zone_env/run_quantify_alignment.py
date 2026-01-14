#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    print("Running zone alignment quantification...")
    
    # Set up environment
    os.environ['PYTHONPATH'] = 'src/'
    
    # Run the analysis script
    cmd = ['python', 'src/quantify_zone_alignment.py']
    print(f"Command: {' '.join(cmd)}")
    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Script completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Script failed with return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 