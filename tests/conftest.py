# tests/conftest.py
import os
import sys

from dotenv import load_dotenv

# project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# load env once
load_dotenv()
