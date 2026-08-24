import os
import sys

# Ensures the project root (where app.py, backend.py, and
# custom_skill_gap_mcp_server.py live) is always importable by tests,
# regardless of how or from where pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))