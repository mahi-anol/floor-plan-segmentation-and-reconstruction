import os
import logging
from datetime import datetime

LOG_FOLDER=f"Training"#{datetime.now().strftime('%H_%M_%S_%d_%m_%Y')}"

log_path=os.path.join(os.getcwd(),"logs",LOG_FOLDER)

os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(log_path,LOG_FOLDER+'.log')

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
