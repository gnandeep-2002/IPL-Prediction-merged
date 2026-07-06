"""
Root conftest.py — ensures the project root is on sys.path so that
`from src.xxx import yyy` works from anywhere the test suite is run.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
