import threading


import numpy as np

from rlive_world.camera.base_camera import BaseCamera
from rlive_world.camera.camera_factory import CameraFactory, CameraConfig
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class CameraService:
    def __init__(self, config: CameraConfig, factory: CameraFactory | None = None):
        self._cfg = config
        self._factory = factory or CameraFactory()
        self._lock = threading.RLock()
        self._cam: BaseCamera | None = None

    def setup(self) -> None:
        with self._lock:
            if self._cam is None:
                logger.info(f"CameraService: starting camera with config {self._cfg}")
                cam = self._factory.build(self._cfg)
                cam.setup()
                self._cam = cam
                logger.info("CameraService: camera started")
            else:
                logger.info("CameraService: camera already started; ignoring start()")

    def get_image(self) -> np.ndarray | None:
        with self._lock:
            if self._cam is None:
                logger.error("CameraService.get_frame: camera is not started")
                return None
            try:
                img = self._cam.get_image()
                # Optional: Validate return type
                if not isinstance(img, np.ndarray):
                    logger.warning("CameraService.get_frame: unexpected image type %s", type(img))
                return img
            except Exception:
                logger.exception("CameraService.get_frame: error obtaining image")
                return None

    def release(self) -> None:
        with self._lock:
            if self._cam is None:
                logger.debug("CameraService: release() called, but camera was not started")
                return
            logger.info("CameraService: stopping camera")
            try:
                self._cam.release()
            except Exception:
                logger.exception("CameraService: error stopping camera")
            finally:
                self._cam = None
                logger.info("CameraService: camera stopped and cleared")

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._cam is not None

    def __del__(self) -> None:
        self.release()

    def __enter__(self) -> "CameraService":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    cfg = CameraConfig(type="dummy", width=320, height=240)
    service = CameraService(config=cfg)

    try:
        service.setup()

        for i in range(3):
            frame = service.get_frame()
            if frame is not None:
                logger.info(f"Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
            else:
                logger.info(f"Frame {i}: Kein Frame erhalten")

        service.release()

    except Exception as e:
        logger.exception(f"Fehler beim Testlauf: {e}")
    finally:
        service.release()
        print("Testlauf abgeschlossen.")
