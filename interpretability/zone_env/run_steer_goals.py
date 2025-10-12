#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("Running goal steering experiment...")
    print("Command: python src/steer_goals.py")
    print("PYTHONPATH: src/")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    
    try:
        # Run the steering script
        result = subprocess.run(
            [sys.executable, 'src/steer_goals.py'],
            env=env,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("Script completed successfully!")
        else:
            print(f"Script failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"Error running script: {e}")

if __name__ == '__main__':
    main() 