from dataclasses import dataclass
from typing import Optional, Callable

from rlive_world.camera import BaseCamera



@dataclass(frozen=True)
class CameraConfig:
    type: str = "dummy"
    id: Optional[int] = 0
    width: int = 640
    height: int = 480



class CameraFactory:
    """
    Camera Factory class, that initializes the camera.
    """
    def __init__(self) -> None:
        # Mapping: kameratyp -> builder-funktion
        self._builders: dict[str, Callable[[CameraConfig], BaseCamera]] = {
            "picam": self._build_pi_camera,
            "webcam": self._build_webcam,
            "dummy": self._build_dummy_camera,  # auto versucht zuerst IDS, fällt sonst auf fake_vid
        }

    def build(self, cfg: CameraConfig) -> BaseCamera:
        cam_type = (cfg.type or "auto").strip().lower()
        # logger.info(f"CameraFactory: building camera type '{cam_type}'")

        builder_func = self._builders.get(cam_type)
        if not builder_func:
            # logger.warning(f"Unknown camera_type='{cam_type}', falling back to 'fake_vid'")
            builder = self._build_dummy_camera

        # Primärer Build-Versuch
        try:
            return builder_func(cfg)
        except Exception:
            # logger.exception(f"Building '{cam_type}' failed.")
            # Fallback-Strategie: bei IDS/auto → fake_vid; bei fake_img → fake_vid; sonst -> raise
            if cam_type in ("ids_peak", "auto", "fake_img"):
                # logger.info("Falling back to 'fake_vid'...")
                return self._build_dummy_camera(cfg)
            raise  # für andere Typen weiterreichen

    @staticmethod
    def _build_dummy_camera(cfg: CameraConfig) -> BaseCamera:
        """Erzeugt eine IDS Peak Kamera, konfiguriert sie aber NICHT startet sie."""
        from dummy_camera import DummyCamera
       return DummyCamera(width=cfg.width, height=cfg.height)

    @staticmethod
    def _build_pi_camera(cfg: CameraConfig) -> BaseCamera:
        """Erzeugt eine IDS Peak Kamera, konfiguriert sie aber NICHT startet sie."""
        from pi_camera import PICamera
        cam = PICamera()
        return cam

    @staticmethod
    def _build_webcam(cfg: CameraConfig) -> BaseCamera:
        """Erzeugt eine IDS Peak Kamera, konfiguriert sie aber NICHT startet sie."""
        from webcam import Webcam
        cam = Webcam(cam_index=cfg.id, width=cfg.width, height=cfg.height)
        return cam
