import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
os.environ["PROJECT_ROOT"] = str(project_root)
os.chdir(project_root)