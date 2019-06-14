# Pass regular log output to devnull; pytest catches log calls by other means
import os
import logging

logging.basicConfig(level=logging.DEBUG, stream=open(os.devnull, "w"))

# Increase log level for certain modules
logging.getLogger("paramspace").setLevel(logging.INFO)
