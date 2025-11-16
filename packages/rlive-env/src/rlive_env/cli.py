from pathlib import Path
from dotenv import load_dotenv


def main():
    ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT / ".env")

    from rlive_env.main_function import main as rlive_env_main

    rlive_env_main()
