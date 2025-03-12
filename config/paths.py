from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# DATA_DIR
SERP_DIR = DATA_DIR / "serp"
DB_DIR = DATA_DIR / "db"
WEBPAGE_DIR = DATA_DIR / "webpage"
OUTPUT_DIR = DATA_DIR / "output"
BROAD_RESEARCH_DIR = OUTPUT_DIR / "broad_research"
DEEP_RESEARCH_DIR = OUTPUT_DIR / "deep_research" / "huawei"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
SERP_DIR.mkdir(exist_ok=True)
WEBPAGE_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
BROAD_RESEARCH_DIR.mkdir(exist_ok=True)
DEEP_RESEARCH_DIR.mkdir(exist_ok=True)

# File paths
CONFIG_FILE = ROOT_DIR / "config.yaml"
LOG_FILE = LOGS_DIR / "remo.log"
