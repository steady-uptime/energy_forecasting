# src/core/utils.py



# Singleton Config Loader
# Since you now have multiple files (config.yaml, logging.yaml, and eventually model.yaml), your Singleton Config loader must be capable of Merging these files into a single configuration object at runtime.
# 
# A professional implementation would look like this:
# 
# Load config.yaml as the base.
# Load logging.yaml and merge it into the base object.
# Load model.yaml and merge it into the base object.
# Apply env_mapping overrides from the OS environment variables.

# Configuration Validation
# In your src/core/utils.py (where your config loader lives), you should implement a Schema Validation step. Before the application starts, the loader should check:
# 
# Do the paths defined in config.yaml actually exist (or can they be created)?
# Are the compute values (like gpu_id) within valid ranges?