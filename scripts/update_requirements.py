# scripts/update_requirements.py
# /scripts/update_requirements.py
import importlib.metadata
import sys
from loguru import logger

# Define the core packages we need to pin for production
CORE_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "pyyaml",
    "loguru",
    "matplotlib",
    "seaborn",
    "pytest"
]

def main():
    logger.info("System: Querying installed package versions for pinning...")
    
    pinned_requirements = []
    
    for package in CORE_PACKAGES:
        try:
            version = importlib.metadata.version(package)
            pinned_requirements.append(f"{package}=={version}")
            logger.info(f"Found {package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            pinned_requirements.append(f"# {package} not installed")
            logger.warning(f"Warning: {package} is not installed in the current environment.")

    print("\n" + "="*30)
    print("COPY THE OUTPUT BELOW")
    print("="*30)
    print("\n".join(pinned_requirements))
    print("="*30)

if __name__ == "__main__":
    main()
