from rlive_world.bolt.boltdummys.dummy_ble_adapter import DummyBleakAdapter, DummyBLEDevice
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# ============================================================
#   DUMMY TOY (replacement for BOLTPLUS)
# ============================================================

class DummyToy:
    """Simplified BOLTPLUS-like object used by DummyFinder."""

    def __init__(self, device: DummyBLEDevice):
        self.name = device.name
        self.address = device.address
        self.adapter = DummyBleakAdapter(self.address)


# ============================================================
#   DUMMY FINDER (replacement for SpheroFinder)
# ============================================================

class DummyFinder:
    def __init__(self):
        self.toys: list[DummyToy] = []
        self.selected_toy: DummyToy | None = None

    def scan_toys(self, **kwargs) -> list[DummyToy]:
        devices = DummyBleakAdapter.scan_toys(**kwargs)
        self.toys = [DummyToy(d) for d in devices]
        logger.debug(f"[DUMMY] Scan result: {[t.name for t in self.toys]}")
        return self.toys

    def select_toy(self, name: str) -> DummyToy | None:
        toy = next((t for t in self.toys if t.name == name), None)
        self.selected_toy = toy
        logger.debug(f"[DUMMY] Selected toy: {toy.name if toy else 'None'}")
        return toy
