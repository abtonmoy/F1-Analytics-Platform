import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # project path
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR/ "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR/ "processed"
    CACHE_DIR = DATA_DIR / "cache"

    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/f1_analytics.db")

    FASTF1_CACHE_DIR = str(CACHE_DIR/ "fastf1")

    SEASONS = [2021, 2022, 2023, 2024]

    @classmethod
    def ensure_directories(cls):
        """create all necessary dirs"""
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

Config.ensure_directories()