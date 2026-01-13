# import os
# import logging
# from datetime import datetime

# LOG_FOLDER=f"Training"#{datetime.now().strftime('%H_%M_%S_%d_%m_%Y')}"

# log_path=os.path.join(os.getcwd(),"logs",LOG_FOLDER)

# os.makedirs(log_path,exist_ok=True)

# LOG_FILE_PATH=os.path.join(log_path,LOG_FOLDER+'.log')

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO,
# )


import os
import logging
from pathlib import Path
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_BASE_DIR = Path(os.getcwd()) / "logs"
LOG_FOLDER = "Training"  # or make it dynamic: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOG_DIR = LOG_BASE_DIR / LOG_FOLDER
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = LOG_DIR / "Training.log"  # consistent name: training.log

# ── Reset & Configure Logging ─────────────────────────────────────────────────
# Remove any existing handlers to prevent duplicates
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Clear the log file at the beginning of each run (very important!)
if LOG_FILE_PATH.exists():
    LOG_FILE_PATH.unlink()  # delete old file
    # or truncate: LOG_FILE_PATH.write_text("")   ← alternative

# Configure fresh logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,               # ← forces reconfiguration even if called multiple times
    encoding='utf-8',
)

# Optional: also print to console for local debugging
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Test log
logging.info("=== NEW TRAINING SESSION STARTED ===")