#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("Running comprehensive steering analysis...")
    print("Command: python src/comprehensive_steering_analysis.py")
    print("PYTHONPATH: src/")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    
    try:
        # Run the comprehensive analysis script
        result = subprocess.run(
            [sys.executable, 'src/comprehensive_steering_analysis.py'],
            env=env,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("Comprehensive analysis completed successfully!")
        else:
            print(f"Analysis failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == '__main__':
    main() 