# Pass regular log output to devnull; pytest catches log calls by other means
import logging
import os

logging.basicConfig(level=logging.DEBUG, stream=open(os.devnull, "w"))

# Adjust log level for certain modules
logging.getLogger("paramspace").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
