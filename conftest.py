"""pytest configuration: make the repository root importable without installing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
