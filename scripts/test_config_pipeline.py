from src.core.config_loader import Config

def main():
    print("Loading config...")
    config = Config.get()

    print("Config loaded successfully.")
    print("Project:", config.project_name)
    print("Version:", config.version)

    # Validate nested configs
    config.data.validate()
    config.training.validate()

    print("Validation passed.")
    print("Data target:", config.data.target_column)
    print("Model:", config.model.name)

if __name__ == "__main__":
    main()
