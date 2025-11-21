import argparse
import uvicorn

from rlive_world.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Entry point for running the world FastAPI server.

    Usage:
      uv run rlive-world          # normal (Pi) mode, host=0.0.0.0
      uv run rlive-world --local  # local (PC) mode, host=127.0.0.1
    """
    parser = argparse.ArgumentParser(description="Run the World FastAPI server.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally (bind to 127.0.0.1 instead of 0.0.0.0).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (DO NOT USE with real hardware!)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug/reload mode.")
    args = parser.parse_args()

    host = cfg.WORLD_HOST
    port = cfg.WORLD_PORT
    debug = args.debug or cfg.DEBUG

    # If --local is used, force localhost binding
    if args.local:
        host = "127.0.0.1"
        logger.info("Running in LOCAL mode (PC server on 127.0.0.1)")

    logger.info(f"Starting World server at http://{host}:{port} (debug={debug}, reload={args.reload})")
    uvicorn.run("rlive_world.world_server:app", host=host, port=port, reload=args.reload, workers=1)


if __name__ == "__main__":
    main()
