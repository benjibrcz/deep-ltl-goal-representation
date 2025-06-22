#!/usr/bin/env python
import os
import subprocess
import sys

def main():
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    
    command = [
        'python', 'src/probe_goal.py'
    ]
    
    print("Running probe_goal.py...")
    print("Command:", ' '.join(command))
    print("PYTHONPATH:", env['PYTHONPATH'])
    
    result = subprocess.run(command, env=env)
    
    if result.returncode == 0:
        print("Script completed successfully!")
    else:
        print(f"Script failed with return code: {result.returncode}")

if __name__ == '__main__':
    main() 