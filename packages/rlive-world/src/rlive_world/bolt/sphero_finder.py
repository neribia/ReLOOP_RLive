from collections.abc import Callable

from sphero_unsw.toy.boltplus import BOLTPLUS

from rlive_common.utils import get_logger

logger = get_logger(__name__)


class SpheroFinder:

    def __init__(
            self,
            scan_fn: Callable[[int], list[BOLTPLUS]] = None,  # Inject BLE scanning function
    ):
        from sphero_unsw import scanner  # keep import internal for easier mocking

        # Default scanning function from library
        self.scan4toys = scan_fn or scanner.find_toys

        self.toys: list[BOLTPLUS] = []
        self.selected_toy: BOLTPLUS | None = None

    def scan_toys(self, scanning_time: int = 3) -> list:
        """Scan for Sphero toys (mockable)."""
        self.toys = self.scan4toys(timeout=scanning_time)

        if not self.toys:
            logger.info("No Sphero toys found.")
        else:
            logger.info("Available Sphero toys:")
            for idx, toy in enumerate(self.toys, start=1):
                logger.info(f"{idx}. {toy.name}")

        return self.toys

    def select_toy(self, name: str) -> BOLTPLUS | None:
        """Select toy by name."""
        self.selected_toy = next((t for t in self.toys if t.name == name), None)

        logger.info(f"Selected toy: {self.selected_toy.name}" if self.selected_toy else f"Toy '{name}' not found.")

        return self.selected_toy

    def get_selected_toy(self) -> object | None:
        """Get the currently selected toy."""
        return self.selected_toy


if __name__ == "__main__":
    finder = SpheroFinder()

    logger.info("Scanning for Sphero robots...")
    toys = finder.scan_toys(scanning_time=3)

    if not toys:
        logger.info("No toys found.")
    else:
        logger.info("\nFound the following toys:")
        for toy in toys:
            logger.info(f" - {toy.name}")

        # Example: select the first toy (or replace with a known name)
        selected = finder.select_toy(toys[0].name)
        logger.info(f"\nSelected toy: {selected}({type(selected)})")
