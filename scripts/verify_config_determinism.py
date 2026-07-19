# scripts/verify_config_determinism.py
import pprint
from src.core.config_loader import config

def verify():
    print("=== DETERMINISTIC CONFIG DUMP ===")
    # We use pprint to see the nested dataclass structure clearly
    pprint.pprint(config)
    
    print("\n=== TYPE INTEGRITY CHECK ===")
    # This proves that 'training' is a TrainingConfig object, not a dict.
    print(f"Type of config.training: {type(config.training)}")
    print(f"Accessing learning_rate: {config.training.learning_rate}")
    
    # This proves we cannot access keys that don't exist in the schema
    try:
        print(config.training.non_existent_key)
    except AttributeError:
        print("Success: Accessing non-existent keys raises AttributeError (Type Safety confirmed).")

if __name__ == "__main__":
    verify()
