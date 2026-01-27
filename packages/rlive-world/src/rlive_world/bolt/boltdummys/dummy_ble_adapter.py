import threading

import asyncio

from rlive_world.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


# FIXME: Create name/address and ID random or from list
class DummyBLEDevice:
    def __init__(self, name="DummyBolt", address="FA:KE:00:00:01"):
        self.name = name
        self.address = address


# ============================================================
#   DUMMY BLEAK ADAPTER (drop-in replacement for BleakAdapter)
# ============================================================

class DummyBleakAdapter:
    @staticmethod
    def scan_toys(timeout: float = cfg.SPHEROBOLTPLUS_SCANNING_TIME):
        """Simulated BLE scan → always returns 1 dummy device."""

        async def fake_scan():
            await asyncio.sleep(timeout)
            return [DummyBLEDevice("DummyBolt", "FA:KE:00:00:01")]

        return asyncio.run(fake_scan())

    @staticmethod
    def scan_toy(name: str):
        async def fake_find():
            if name == "DummyBolt":
                return DummyBLEDevice("DummyBolt", "FA:KE:00:00:01")
            return None

        return asyncio.run(fake_find())

    # ------------------------------------------------------------

    def __init__(self, address: str):
        self.address = address
        self.connected = False

        # event loop thread simulating bleak structure
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

        self._execute(self._connect())

    async def _connect(self):
        await asyncio.sleep(0.2)
        self.connected = True
        logger.debug(f"[DUMMY] Connected to {self.address}")

    def _execute(self, coro):
        with self._lock:
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ------------------------------------------------------------

    def write(self, uuid: str, data: bytes):
        async def fake_write():
            await asyncio.sleep(0.05)
            logger.debug(f"[DUMMY] Write to {uuid}: {data}")

        return self._execute(fake_write())

    def set_callback(self, uuid: str, cb):
        async def fake_callback():
            logger.debug(f"[DUMMY] Notifications enabled for {uuid}")
            await asyncio.sleep(0.3)

        return self._execute(fake_callback())

    def close(self, disconnect=True):
        async def fake_disc():
            await asyncio.sleep(0.1)
            logger.debug(f"[DUMMY] Disconnected from {self.address}")

        if disconnect:
            self._execute(fake_disc())

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
