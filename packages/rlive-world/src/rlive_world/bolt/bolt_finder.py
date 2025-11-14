from sphero_unsw import scanner
from sphero_unsw.toy.boltplus import BOLTPLUS
import inspect

from rlive_common.utils import get_logger

logger = get_logger(__name__)


class SpheroFinder:
    # Constructor: initializes the object with empty toy list and no selected toy
    def __init__(self):
        self.toys = []  # This will hold the list of found Sphero toys
        self.selected_toy = None  # This will store the toy selected by the user

    def scan_toys(self, scanning_time=3):
        self.toys = scanner.find_toys(timeout=scanning_time)

        # Check if any toys were found
        if not self.toys:
            # No toys found, print a warning
            logger.info("No Sphero toys found. Ensure your Bluetooth is on and the toy is awake.")
        else:
            # Toys were found, print them in a numbered list
            logger.info("Available Sphero toys:")
            for idx, toy in enumerate(self.toys, start=1):
                logger.info(f"{idx}. {toy.name}")  # Show index and toy name

    def select_toy(self, name: str) -> BOLTPLUS:
        """
        Return the toy with the given name, or None if not found.
        """
        self.selected_toy = next((t for t in self.toys if t.name == name), None)

        logger.info(f"Selected toy: {self.selected_toy.name}" if self.selected_toy else f"Toy '{name}' not found.")

        return self.selected_toy

    # Function to get the selected toy
    def get_selected_toy(self):
        """
        Get the currently selected toy.

        Returns:
            object: The selected toy object, or None if no toy is selected.
        """
        return self.selected_toy

    @staticmethod
    def inspect_object(obj):
        print("=== Methods ===")
        for name, member in inspect.getmembers(obj, inspect.ismethod):
            print(name)

        print("\n=== Attributes ===")
        for name, val in inspect.getmembers(obj, lambda x: not (inspect.ismethod(x))):
            if not name.startswith("__"):
                print(name, "=", val)


if __name__ == "__main__":
    finder = SpheroFinder()
    finder.scan_toys()
    finder.select_toy("BP-D217")
    logger.info(finder.get_selected_toy())

