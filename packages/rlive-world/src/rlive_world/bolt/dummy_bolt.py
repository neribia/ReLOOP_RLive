import asyncio
import random
import time

from rlive_world.bolt.base_robot import BaseRobot
from rlive_world.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class FakeBLEDevice:
    """A very small placeholder to mimic a BLE device."""

    def __init__(self, name="DummyBolt", address="FA:KE:DE:VI:CE"):
        self.name = name
        self.address = address


class DummyBolt(BaseRobot):
    """
    A realistic robot simulator that mimics BLE behavior.
    """

    def __init__(self):
        logger.debug("Initializing DummyBolt class")
        self.connected = False
        self.device = None
        self.name = None


        self.heading = 0  # degrees
        self.move_delay = 0.1  # seconds per move

    # --------------------------------------------------------------------------
    # Fake BLE scanning + connection
    # --------------------------------------------------------------------------

    def connect(self):
        logger.info("DummyBolt: scanning for devices...")

        async def fake_scan():
            await asyncio.sleep(5)  # Simulate BLE scan
            return FakeBLEDevice()

        # mimic bleak behavior: scanning → connecting
        device = asyncio.run(fake_scan())
        self.device = device

        logger.info(f"DummyBolt: found device {device.name} ({device.address})")

        async def fake_connect():
            await asyncio.sleep(0.3)  # Simulate connection handshake
            return True

        logger.info("DummyBolt: connecting...")
        result = asyncio.run(fake_connect())

        if result:
            logger.info("DummyBolt connected successfully.")
            self.connected = True
        else:
            logger.error("DummyBolt failed to connect.")
            self.connected = False

        return self.connected

    # --------------------------------------------------------------------------
    # Movement simulation
    # --------------------------------------------------------------------------

    def move(self, heading, speed=cfg.SPHEROBOLTPLUS_SPEED, duration=cfg.SPHEROBOLTPLUS_DURATION):
        if not self.connected:
            logger.warning("DummyBolt move() called but robot is not connected.")
            return

        async def fake_move():
            await asyncio.sleep(duration * 2)

        asyncio.run(fake_move())

        logger.info(f"Moving Fake: heading={heading}, speed={speed}, duration={duration}")

    # --------------------------------------------------------------------------
    # Fake disconnect
    # --------------------------------------------------------------------------

    def disconnect(self):
        if not self.connected:
            logger.debug("DummyBolt: already disconnected.")
            return

        logger.debug(f"DummyBolt: disconnecting from {self.device.address}...")

        async def fake_disconnect():
            await asyncio.sleep(0.2)

        asyncio.run(fake_disconnect())

        self.connected = False
        logger.info("DummyBolt disconnected.")
