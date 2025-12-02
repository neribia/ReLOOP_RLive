from typing import Optional, Any

from rlive_world.camera.base_camera import BaseCamera
from rlive_world.camera.camera_factory import CameraFactory, CameraConfig
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class CameraService:
    def __init__(self, config: CameraConfig, factory: Optional[CameraFactory] = None):
        self._cfg = config
        self._factory = factory or CameraFactory()
        self._cam: Optional[BaseCamera] = None

    def setup(self) -> None:
        if self._cam is None:
            logger.info(f"CameraService: starting camera with config {self._cfg}")
            cam = self._factory.build(self._cfg)
            cam.setup()
            self._cam = cam
            logger.info("CameraService: camera started")
        else:
            logger.info("CameraService: camera already started; ignoring start()")

    def get_frame(self) -> Any:
        cam = self._cam
        if cam is None:
            logger.error("CameraService.get_frame: camera is not started")
            return None
        return cam.get_image()

    def release(self) -> None:
        if self._cam:
            try:
                logger.info("CameraService: stopping camera")
                self._cam.release()
            except Exception as e:
                logger.exception(f"Error stopping camera: {e}")
            finally:
                self._cam = None
                logger.info("CameraService: camera stopped and cleared")
        else:
            logger.info("CameraService: stop() called, but camera was not started")


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
        print("Testlauf abgeschlossen.")
