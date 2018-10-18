# Pass regular log output to devnull; pytest catches log calls by other means
import os
import logging
logging.basicConfig(level=logging.INFO, stream=open(os.devnull, "w"))
