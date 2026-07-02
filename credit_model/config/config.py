from pathlib import Path
from typing import List
from pydantic import BaseModel
import yaml

PACKAGE_ROOT = Path(__file__).parent.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config" / "config.yml"

class AppConfig(BaseModel):
    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    data_folder: str
    train_data_file: str
    credit_record_file: str
    test_size: float
    random_state: int
    target: str
    features_to_drop: List[str]
    numerical_features: List[str]
    categorical_features: List[str]

def fetch_config_from_yaml() -> AppConfig:
    with open(CONFIG_FILE_PATH, "r") as f:
        parsed = yaml.safe_load(f)
    return AppConfig(**parsed)

config = fetch_config_from_yaml()