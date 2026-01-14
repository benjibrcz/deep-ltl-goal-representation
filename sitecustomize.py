import os, sys
root = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(root, "src")
if os.path.isdir(src) and src not in sys.path:
    sys.path.insert(0, src)
