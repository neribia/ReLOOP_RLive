"""World loading strategies for SAPIEN integration.

This module provides pluggable strategies for loading complete worlds
(environment + robot) in SAPIEN simulations. Each strategy implements 
a consistent interface for loading and initializing worlds.

The module uses an Enum + Registry pattern for type-safe and extensible
strategy management.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any
import numpy as np

from rlive_common.utils import get_logger

logger = get_logger(__name__)

try:
    import sapien
except ImportError:
    sapien = None

# --- SAPIEN WINDOWS PATCH START ---
# Fix for missing pinocchio on Windows causing Viewer crashes
if sapien is not None:
    try:
        from sapien.wrapper import pinocchio_model
        if pinocchio_model.PinocchioModel is None:
            logger.warning("⚠️ Pinocchio not found. Patching SAPIEN PinocchioModel to prevent Viewer crashes.")
            
            class DummyPinocchioModel:
                def __init__(self, xml_string, gravity_vec):
                    pass
                
                def set_link_order(self, order):
                    pass
                
                def set_joint_order(self, order):
                    pass
                
                def compute_inverse_kinematics(self, link_index, pose, initial_qpos, 
                                             active_qmask=None, max_iterations=100, 
                                             eps=1e-4, damp=1e-6):
                    # Return result (same qpos), success (False), error (High)
                    return initial_qpos, False, 1.0
                
                def compute_forward_kinematics(self, qpos):
                    pass

            pinocchio_model.PinocchioModel = DummyPinocchioModel
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Warning: Failed to patch SAPIEN PinocchioModel: {e}")
# --- SAPIEN WINDOWS PATCH END ---

from rlive_sim.config import RESOURCES_DIR
from rlive_sim.config.bolt_config import BOLT_DEFAULTS, BoltDefaults
from rlive_sim.config.eurobox_config import EUROBOX_DEFAULTS, EuroBoxDefaults
from rlive_sim.engine.sapien.sapianrobot import SapianRobot, SimpleSphereRobot, KinematicSpheroRobot

SAPIAN_DIR = RESOURCES_DIR / "sapian"


# ============================================================================
# ENUM FOR WORLD LOADING TYPES
# ============================================================================

class WorldLoaderType(str, Enum):
    """Enumeration of available world loading strategies.
    
    This enum provides type-safe access to all registered world loaders.
    """
    
    SAPIEN = "sapien"
    """Programmatically built rigid sphere with ground plane (fastest, default)."""
    
    GLB = "glb"
    """Load from GLB/GLTF file with ground plane or as world (resources/sapian/bolt_shell.glb or euro_box.glb)."""

    GLB_FLAT = "glb_flat"
    """KinematicSpheroRobot on a flat ground plane (no euro box). Used for calibration."""

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> "WorldLoaderType":
        """Create enum from string value.
        
        Args:
            value: String value ('sapien' or 'glb')
            
        Returns:
            WorldLoaderType enum value
            
        Raises:
            ValueError: If value is not a valid world loader type
        """
        for member in cls:
            if member.value == value:
                return member
        
        valid = [member.value for member in cls]
        raise ValueError(
            f"Invalid world loader type '{value}'. "
            f"Must be one of {valid}"
        )


# ============================================================================
# REGISTRY FOR WORLD LOADING STRATEGIES
# ============================================================================

_WORLD_LOADER_REGISTRY: Dict[WorldLoaderType, type["WorldLoadingStrategy"]] = {}


def register_world_loader(loader_type: WorldLoaderType):
    """Decorator to register a world loading strategy.
    
    This decorator registers a WorldLoadingStrategy implementation for
    a specific WorldLoaderType enum value.
    
    Example:
        @register_world_loader(WorldLoaderType.SAPIEN)
        class SapienWorldStrategy(WorldLoadingStrategy):
            def load(self, scene, config):
                # implementation
                pass
    
    Args:
        loader_type: The WorldLoaderType enum value this strategy handles
        
    Returns:
        Decorator function
    """
    def decorator(cls: type["WorldLoadingStrategy"]) -> type["WorldLoadingStrategy"]:
        _WORLD_LOADER_REGISTRY[loader_type] = cls
        return cls
    
    return decorator


class WorldLoadingStrategy(ABC):
    """Base class for world loading strategies.
    
    Strategies load complete worlds including environment and robot.
    All strategies must return compatible objects that support:
    - .pose property
    - .set_pose() method
    - .get_linear_velocity() method
    - .get_angular_velocity() method
    - .add_force_at_point() method
    - .find_component_by_type() method (for physics access)
    """
    
    @abstractmethod
    def load(
        self,
        scene: "sapien.Scene",
        config: Dict[str, Any]
    ) -> SapianRobot:
        """Load world and robot, return robot objects.
        
        The strategy is responsible for:
        - Loading the environment (ground plane or GLB model)
        - Loading the robot
        - Setting up the scene
        
        Args:
            scene: SAPIEN scene to load world into
            config: Configuration dictionary with keys:
                - robot_radius: Robot radius in meters
                - robot_mass: Robot mass in kg
                - friction: Friction coefficient
                - restitution: Restitution coefficient
                - robot_path: Optional custom path to robot file
        
        Returns:
            The loaded robot as a Robot instance.
            The control link (shell) should be discoverable (e.g. named "shell" or root).
        
        Raises:
            FileNotFoundError: If required files are not found
            RuntimeError: If loading fails
        """
        pass


@register_world_loader(WorldLoaderType.SAPIEN)
class SapienWorldStrategy(WorldLoadingStrategy):
    """Load world with programmatically built rigid sphere robot.
    
    This is the fastest strategy, ideal for simulation-heavy workloads
    where visual fidelity is not critical.
    """
    
    def load(
        self,
        scene: "sapien.Scene",
        config: Dict[str, Any]
    ) -> SapianRobot:
        """Build sphere programmatically with ground plane.
        
        Args:
            scene: SAPIEN scene
            config: Configuration dictionary
            
        Returns:
            The robot Articulation.
        """
        if sapien is None:
            raise ImportError("SAPIEN is not installed")
        
        # Load ground plane
        scene.add_ground(altitude=0)

        robot = SimpleSphereRobot(scene, config)
        robot.load()
        
        return robot
    

@register_world_loader(WorldLoaderType.GLB)
class GLBWorldStrategy(WorldLoadingStrategy):
    """Load world from GLB/GLTF file with sphere collision robot.
    
    Loads visual geometry from GLB file but uses sphere collision
    for reliable physics simulation. Supports:
    - bolt_shell.glb: Robot shell with ground plane
    - euro_box.glb: Euro box world (no ground plane needed)
    - spheroboltplus_robot.glb: Robot with robot and ground plane
    """
    
    def load(
        self,
        scene: "sapien.Scene",
        config: Dict[str, Any]
    ) -> SapianRobot:
        """Load complete world with Euro Box, Bolt Shell and robot.
        
        Loads:
        1. euro_box.glb (Static environment)
        2. bolt_shell.glb (Robot Shell)
        3. spheroboltplus_robot.glb (Robot robot)
        
        Args:
            scene: SAPIEN scene
            config: Configuration dictionary
            
        Returns:
            The robot Articulation.
        """
        if sapien is None:
            raise ImportError("SAPIEN is not installed")

        # Load ground plane (as a safety floor)
        scene.add_ground(altitude=0)
        
        # 1. Load Euro Box Environment (Static)
        self._load_euro_box_env(scene, config)
        
        # 2. Load Robot based on configuration        
        return self._load_robot(scene, config)

    def _load_euro_box_env(self, scene: "sapien.Scene", config: Dict[str, Any]) -> None:
        """Load euro box as static world environment."""
        glb_path = SAPIAN_DIR / "euro_box.glb"
        
        if not glb_path.exists():
            logger.warning(f"⚠ Warning: Euro box GLB not found at {glb_path}")
            return

        eurobox_cfg: EuroBoxDefaults = config.get('eurobox_config', EUROBOX_DEFAULTS)
        env_mat = scene.create_physical_material(
            static_friction=eurobox_cfg.static_friction,
            dynamic_friction=eurobox_cfg.dynamic_friction,
            restitution=eurobox_cfg.restitution,
        )

        builder = scene.create_actor_builder()
        
        # Visual
        builder.add_visual_from_file(str(glb_path))
        
        try:
            # Try SAPIEN 3 naming first
            builder.add_nonconvex_collision_from_file(str(glb_path), material=env_mat)
        except AttributeError:
            logger.warning(f"⚠ Warning: Could not create mesh collision for euro box")
        
        box = builder.build_static(name="euro_box")
        box.set_pose(sapien.Pose([0, 0, 0]) )

    def _load_robot(self, scene: "sapien.Scene", config: Dict[str, Any]) -> SapianRobot:
        """Load robot with Kinematic Visuals."""
        shell_path = str(SAPIAN_DIR / "spheroboltplus_shell.glb")
        robot_path = str(SAPIAN_DIR / "spheroboltplus_robot_simple.glb")

        robot_instance = KinematicSpheroRobot(scene, config, shell_path, robot_path)
        robot_instance.load()

        return robot_instance


@register_world_loader(WorldLoaderType.GLB_FLAT)
class GLBFlatWorldStrategy(WorldLoadingStrategy):
    """KinematicSpheroRobot on a flat ground plane — no euro box.

    Identical to GLBWorldStrategy but skips the euro_box.glb load.
    The ground plane uses the friction/restitution from config so the
    physics match what the real euro box floor produces.

    Use this for calibration runs where you want the same robot but an
    unbounded flat ground.
    """

    def load(self, scene: "sapien.Scene", config: Dict[str, Any]) -> SapianRobot:
        if sapien is None:
            raise ImportError("SAPIEN is not installed")

        eurobox_cfg: EuroBoxDefaults = config.get('eurobox_config', EUROBOX_DEFAULTS)
        ground_mat = scene.create_physical_material(
            static_friction=eurobox_cfg.static_friction,
            dynamic_friction=eurobox_cfg.dynamic_friction,
            restitution=eurobox_cfg.restitution,
        )
        scene.add_ground(altitude=0, material=ground_mat)

        shell_path = str(SAPIAN_DIR / "spheroboltplus_shell.glb")
        robot_path = str(SAPIAN_DIR / "spheroboltplus_robot_simple.glb")
        robot_instance = KinematicSpheroRobot(scene, config, shell_path, robot_path)
        robot_instance.load()
        
        return robot_instance


class WorldLoaderFactory:
    """Factory for creating appropriate world loading strategies.
    
    Uses the Registry pattern to manage strategy registration and creation.
    Strategies are automatically registered via @register_world_loader decorator.
    """
    
    @classmethod
    def create(cls, world_type: WorldLoaderType | str) -> WorldLoadingStrategy:
        """Create a loading strategy for the given type.
        
        Args:
            world_type: Either WorldLoaderType enum or string value
                       ('sapien', 'glb', or custom registered type)
            
        Returns:
            WorldLoadingStrategy instance
            
        Raises:
            ValueError: If world_type is not registered
            TypeError: If world_type is not WorldLoaderType enum or string
        """
        # Convert string to enum if necessary
        if isinstance(world_type, str):
            try:
                world_type = WorldLoaderType.from_string(world_type)
            except ValueError as e:
                valid = list(_WORLD_LOADER_REGISTRY.keys())
                raise ValueError(
                    f"Invalid world_type '{world_type}'. "
                    f"Must be one of {[t.value for t in valid]} or a registered custom type"
                ) from e
        
        if not isinstance(world_type, WorldLoaderType):
            raise TypeError(
                f"world_type must be WorldLoaderType enum or string, "
                f"got {type(world_type).__name__}"
            )
        
        if world_type not in _WORLD_LOADER_REGISTRY:
            valid = list(_WORLD_LOADER_REGISTRY.keys())
            raise ValueError(
                f"World loader type {world_type} is not registered. "
                f"Available types: {valid}"
            )
        
        strategy_class = _WORLD_LOADER_REGISTRY[world_type]
        return strategy_class()
    
    @classmethod
    def register(
        cls,
        loader_type: WorldLoaderType,
        strategy_class: type[WorldLoadingStrategy]
    ) -> None:
        """Register a world loading strategy."""
        if not issubclass(strategy_class, WorldLoadingStrategy):
            raise TypeError(
                f"Strategy class must inherit from WorldLoadingStrategy, "
                f"got {strategy_class}"
            )
        
        if loader_type in _WORLD_LOADER_REGISTRY:
            raise ValueError(
                f"World loader type {loader_type} is already registered. "
                f"Cannot override with {strategy_class}"
            )
        
        _WORLD_LOADER_REGISTRY[loader_type] = strategy_class
    
    @classmethod
    def get_available_types(cls) -> list[WorldLoaderType]:
        """Get list of available strategy types."""
        return list(_WORLD_LOADER_REGISTRY.keys())
    
    @classmethod
    def get_available_type_names(cls) -> list[str]:
        """Get list of available strategy type names (string values)."""
        return [t.value for t in _WORLD_LOADER_REGISTRY.keys()]
    
    @classmethod
    def is_available(cls, world_type: WorldLoaderType | str) -> bool:
        """Check if a world loader type is available."""
        if isinstance(world_type, str):
            try:
                world_type = WorldLoaderType.from_string(world_type)
            except ValueError:
                return False
        
        return world_type in _WORLD_LOADER_REGISTRY


if __name__ == "__main__":
    from rlive_sim.utils import deg_to_rad
    import argparse

    # Configuration
    DEFAULT_CONFIG = {
        'robot_radius': 0.0365,
        'robot_mass': 0.12,
        'friction': 0.8,
        'restitution': 0.1,
    }

    parser = argparse.ArgumentParser(description="Test World Loaders")
    parser.add_argument("--loader", default="glb", choices=["sapien", "glb"], help="Loader type")
    parser.add_argument("--mechanics", default="pendulum", choices=["pendulum", "idealized"], help="Robot mechanics type (only for GLB)")
    args = parser.parse_args()

    # Update config
    DEFAULT_CONFIG['robot_mechanics'] = args.mechanics

    # ['sapien', 'glb']
    loader_type = WorldLoaderType.from_string(args.loader)

    logger.info(f"Testing WorldLoader: {loader_type}")

    # Create scene
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)

    # Load world using factory
    loader = WorldLoaderFactory.create(loader_type)
    robot = loader.load(scene, DEFAULT_CONFIG)

    logger.info(f"✓ Loaded world with robot: {robot}")

    # For testing, find shell link
    shell_link = None
    if hasattr(robot, "shell_link"):
        shell_link = robot.shell_link
    elif hasattr(robot, "get_links"):
        for link in robot.get_links():
            if link.name == "shell":
                shell_link = link
                break
        if shell_link is None and len(robot.get_links()) > 0:
            shell_link = robot.get_links()[0]

    logger.info(f"✓ Shell link (Control Link): {shell_link}")


    # Create viewer
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=0, y=-1.0, z=1.0)
    viewer.set_camera_rpy(r=0, p=deg_to_rad(-45), y=deg_to_rad(-90))
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    # Setup lighting
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # Simulation loop
    steps = 0
    logger.info("\nSimulation running... (press 'q' to quit)")
    logger.info("Controls: I (Forward), K (Backward), J (Left), L (Right)")

    robot_pose = shell_link.pose if shell_link else None
    logger.info(f"Robot pose: {robot_pose}")

    
    while not viewer.closed:
        # Simple Movement Control for Testing
        if shell_link:
            force = np.zeros(3)
            torque = np.zeros(3)
            # Adjust force magnitude as needed, assuming mass ~0.12kg
            force_mag = 0.5 
            
            if viewer.window.key_down('i'): # +X
                force[0] += force_mag
            if viewer.window.key_down('k'): # -X
                force[0] -= force_mag
            if viewer.window.key_down('j'): # +Y
                force[1] += force_mag
            if viewer.window.key_down('l'): # -Y
                force[1] -= force_mag
                
            # Apply force to the shell center
            # Note: shell_link is a PhysxArticulationLinkComponent
            shell_link.add_force_torque(force, torque)

        scene.step()
        scene.update_render()
        viewer.render()

        if steps % 500 == 0:
            robot_pose = shell_link.pose if shell_link else None
            logger.info(f"Step {steps}: Robot pose = {robot_pose}")


        steps += 1

