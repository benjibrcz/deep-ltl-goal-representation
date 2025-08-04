#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("Running improved zone probing analysis...")
    print("Command: python src/improved_zone_probing.py")
    print("PYTHONPATH: src/")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    
    try:
        # Run the improved zone probing script
        result = subprocess.run(
            [sys.executable, 'src/improved_zone_probing.py'],
            env=env,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("Improved zone probing analysis completed successfully!")
        else:
            print(f"Analysis failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == '__main__':
    main() 