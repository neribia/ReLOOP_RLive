from dataclasses import dataclass
from typing import Optional, Callable

from rlive_world.camera.base_camera import BaseCamera

from rlive_common.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    """
    Configuration dataclass for camera initialization.

    Attributes:
        type (str): The type of camera to be created (e.g., 'picam', 'webcam', 'dummy').
        id (Optional[int]): The camera index or ID (e.g., for webcam devices). Defaults to 0.
        width (int): The desired camera frame width in pixels. Defaults to 640.
        height (int): The desired camera frame height in pixels. Defaults to 480.
    """

    type: str = "dummy"
    id: Optional[int] = 0
    width: int = 640
    height: int = 480


class CameraFactory:
    """
    Factory class responsible for creating and initializing camera instances.

    The factory provides an abstraction layer to create different types of camera
    objects (e.g., PiCamera, Webcam, DummyCamera) based on the given configuration.

    Example:
        # >>> cfg = CameraConfig(type="webcam", id=0, width=1280, height=720)
        # >>> factory = CameraFactory()
        # >>> camera = factory.build(cfg)
    """

    def __init__(self) -> None:
        """
        Initializes the CameraFactory and registers available camera builder functions.
        """
        self._builders: dict[str, Callable[[CameraConfig], BaseCamera]] = {
            "picam": self._build_pi_camera,
            "webcam": self._build_webcam,
            "dummy": self._build_dummy_camera,  # Default fallback
        }

    def build(self, cfg: CameraConfig) -> BaseCamera:
        """
        Builds and returns a camera instance based on the given configuration.

        If the specified camera type cannot be created, the factory automatically
        falls back to a dummy camera (fake video feed).

        Args:
            cfg (CameraConfig): The configuration object specifying the camera type and parameters.

        Returns:
            BaseCamera: An instance of the requested camera type.

        Raises:
            Exception: If camera creation fails for unsupported types and fallback is not possible.
        """
        cam_type = (cfg.type or "dummy").strip().lower()
        logger.info(f"Building camera of type '{cam_type}'...")

        builder_func = self._builders.get(cam_type, self._build_dummy_camera)

        if builder_func is self._build_dummy_camera and cam_type != "dummy":
            logger.warning(f"Unknown camera type '{cam_type}', falling back to 'dummy'")

        try:
            return builder_func(cfg)
        except Exception as e:
            logger.exception(f"Failed to build camera type '{cam_type}': {e}")
            if cam_type in {"picam", "webcam"}:
                # logger.info("Falling back to 'dummy'...")
                return self._build_dummy_camera(cfg)
            raise RuntimeError(f"Camera type '{cam_type}' could not be built") from e

    @staticmethod
    def _build_dummy_camera(cfg: CameraConfig) -> BaseCamera:
        """
        Creates a dummy (fake) camera instance, used as fallback when real cameras are unavailable.

        Args:
            cfg (CameraConfig): Camera configuration object.

        Returns:
            BaseCamera: A dummy camera instance.
        """
        from rlive_world.camera.dummy_camera import DummyCamera

        return DummyCamera(width=cfg.width, height=cfg.height)

    @staticmethod
    def _build_pi_camera(cfg: CameraConfig) -> BaseCamera:
        """
        Creates a Raspberry Pi camera instance (e.g., using the PiCamera module).

        Args:
            cfg (CameraConfig): Camera configuration object.

        Returns:
            BaseCamera: A PiCamera instance.
        """
        from rlive_world.camera.pi_camera import PICamera

        return PICamera()

    @staticmethod
    def _build_webcam(cfg: CameraConfig) -> BaseCamera:
        """
        Creates a webcam instance using a connected USB or built-in camera.

        Args:
            cfg (CameraConfig): Camera configuration object.

        Returns:
            BaseCamera: A Webcam instance.
        """
        from rlive_world.camera.webcam import Webcam

        return Webcam(cam_index=cfg.id, width=cfg.width, height=cfg.height)
