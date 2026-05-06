"""SAPIEN Integrated Engine implementation.

This module provides an integrated simulation engine using SAPIEN.
It implements the Sphero BOLT+ robot with multiple loading options:
- rigid_body: Programmatically built sphere
- urdf: URDF file loaded robot
- glb: GLB/GLTF file loaded robot
"""
import warnings
from typing import Any

import numpy as np

try:
    import sapien
except ImportError:
    sapien = None

from rlive_sim.engine.core.base_integrated_engine import BaseIntegratedEngine
from rlive_sim.engine.core.base_physics_engine import PhysicsState
from rlive_sim.engine.core.registry import register_integrated_backend
from rlive_sim.config import IntegratedBackend, SAPIEN_DEFAULTS
from rlive_sim.config.bolt_config import BOLT_DEFAULTS, BoltDefaults
from rlive_sim.config.eurobox_config import EUROBOX_DEFAULTS, EuroBoxDefaults
from rlive_sim.utils.math_utils import euler_to_quat, deg_to_rad


from rlive_sim.engine.sapien.sphero_controller import SpheroController, Phase
from rlive_sim.engine.sapien.world_loading import WorldLoaderFactory, WorldLoaderType

@register_integrated_backend(IntegratedBackend.SAPIEN)
class SapienIntegratedEngine(BaseIntegratedEngine):
    """SAPIEN integrated engine for Sphero simulation.

    Supports multiple world loading options:
    - 'sapien': Simple ground plane with programmatically built sphere (default)
    - 'urdf': Load from URDF file with ground plane
    - 'glb': Load from GLB/GLTF file (supports euro_box without ground plane)
    """

    def __init__(
        self,
        gravity: list[float] | None = None,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
        robot_type: str = SAPIEN_DEFAULTS.robot_type, # Keeps name for config compatibility, but maps to world_type
        robot_path: str | None = SAPIEN_DEFAULTS.robot_path,
        max_speed_ms: float = SAPIEN_DEFAULTS.max_speed_ms,
        controller_kp: float = SAPIEN_DEFAULTS.controller_kp,
        controller_kd: float = SAPIEN_DEFAULTS.controller_kd,
        robot_radius: float = SAPIEN_DEFAULTS.robot_radius, # FIXME: Use Bolt_Default
        robot_mass: float = SAPIEN_DEFAULTS.robot_mass,  # FIXME: Use Bolt_Default
        bolt_config: BoltDefaults = BOLT_DEFAULTS,
        eurobox_config: EuroBoxDefaults = EUROBOX_DEFAULTS,
        sim_dt: float = SAPIEN_DEFAULTS.sim_dt,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize SAPIEN Integrated Engine.
        
        Args:
            gravity: Gravity vector [x, y, z]
            width: Render width in pixels
            height: Render height in pixels
            channels: Number of render channels
            robot_type: World loading type ('sapien', 'urdf', or 'glb') - formerly robot_type
            robot_path: Optional custom path to robot/world file
            max_speed_ms: Maximum speed in m/s
            controller_kp: Controller proportional gain
            controller_kd: Controller derivative gain
            robot_radius: Robot radius in meters
            robot_mass: Robot mass in kg
            bolt_config: Bolt physics configuration (friction, restitution, etc.)
            eurobox_config: EuroBox physics configuration (friction, restitution, etc.)
            sim_dt: Physics timestep in seconds
            *args: Unsupported positional arguments
            **kwargs: Unsupported keyword arguments
        
        Warning:
            Any arguments in *args and **kwargs are not supported and will be ignored.
        """
        if sapien is None:
            raise ImportError("SAPIEN is not installed. Please install it with `pip install sapien`.")
        
        # Extract supported kwargs
        self.use_viewer = kwargs.pop("use_viewer", False)

        # Warn about unsupported args
        if args or kwargs:
            warnings.warn(
                f"Unsupported arguments passed to SapienIntegratedEngine.__init__(): "
                f"args={args}, kwargs={kwargs}. These will be ignored.",
                UserWarning,
                stacklevel=2
            )

        super().__init__(gravity=gravity, width=width, height=height, channels=channels)
        
        # Validate robot_type (now maps to world_type)
        try:
            # Map legacy 'rigid_body' to 'sapien' if needed, though config should be updated
            if robot_type == "rigid_body":
                robot_type = "sapien"
                
            if isinstance(robot_type, str):
                WorldLoaderType.from_string(robot_type)
        except ValueError as e:
            raise ValueError(str(e)) from e
        
        # Store configuration as private attributes
        self._robot_type = robot_type
        self._robot_path = robot_path
        self._max_speed_ms = max_speed_ms
        self._controller_kp = controller_kp
        self._controller_kd = controller_kd
        self._robot_radius = robot_radius
        self._robot_mass = robot_mass
        self._bolt_config = bolt_config
        self._eurobox_config = eurobox_config

        self.sim_dt = sim_dt
        self.render_dt = 1.0 / 60.0
        
        # Viewer configuration
        # self.use_viewer is already set above
        self.viewer = None
        self.render_stride = 1

        # Initialize SAPIEN Scene
        self.scene = sapien.Scene()
        if gravity:
            self.scene.gravity = gravity
        else:
            self.scene.gravity = [0, 0, -9.81]
        self.scene.timestep = self.sim_dt

        # Robot
        self.robot = None
        self.shell_link = None
        self.sled_link = None

        # Setup Viewer
        self._setup_viewer()
        
        # Initial render to populate scene
        self.scene.update_render()
        if self.viewer:
            self.viewer.render()

        # Setup Controller
        self.controller = SpheroController(
            max_speed_ms=self._max_speed_ms,
            # max_accel_ms2=self._max_accel_ms2, # TODO: Setup
            kp=self._controller_kp,
            kd=self._controller_kd,
            mass=self._robot_mass
        )

    def _setup_world(self) -> tuple:
        """Load complete world (environment + robot) using strategy.
        
        Uses WorldLoaderFactory to create the correct loader based on
        configuration. The strategy handles both ground setup (if needed)
        and robot loading.
        
        Returns:
            Tuple of (robot, shell_link, sled_link) where:
            - robot: Main robot object (Robot instance)
            - shell_link: The controlled link (Actor-compatible)
            - sled_link: Alias for shell_link
        """
        # Prepare configuration dict for loader
        loader_config = {
            'robot_type': self._robot_type,
            'robot_path': self._robot_path,
            'robot_radius': self._robot_radius,
            'robot_mass': self._robot_mass,
            'bolt_config': self._bolt_config,
            'eurobox_config': self._eurobox_config,
        }

        # Setup Camera
        self._setup_camera()

        # Setup Lights
        self._setup_light()
        
        # Create and execute loader using factory
        loader = WorldLoaderFactory.create(self._robot_type)
        robot = loader.load(self.scene, loader_config)
        
        # In the new abstraction, the returned robot is an instance of Robot
        shell_link = robot.shell_link
        
        # sled_link is alias for shell_link
        sled_link = shell_link
        
        return robot, shell_link, sled_link

    def _setup_viewer(self) -> None:
        """Setup SAPIEN viewer if requested."""
        if not self.use_viewer:
            self.viewer = None
            return

        # Render stride to target ~60FPS in viewer if enabled
        self.render_stride = int(self.render_dt / self.sim_dt)
        if self.render_stride < 1: self.render_stride = 1

            # Always render every step if viewer is active to prevent freezing/white screen
            # render_stride = 1

        try:
            self.viewer = self.scene.create_viewer()
            self.viewer.set_camera_xyz(x=0, y=-1.0, z=1.0)
            self.viewer.set_camera_rpy(r=0, p=deg_to_rad(-45), y=deg_to_rad(-90))
            self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        except Exception as e:
             warnings.warn(f"Failed to create SAPIEN viewer: {e}")
             self.viewer = None

    def _setup_camera(self, width: int = 640, height: int = 480, near: float = 0.1, far: float = 100.) -> None:
        """Setup camera in sapian.
        Source: https://sapien-sim.github.io/docs/user_guide/rendering/camera.html
        """
        # Compute the camera pose by specifying forward(x), left(y) and up(z)
        cam_pos = np.array([-0.5, 0, 1])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos

        self.camera = self.scene.add_camera(
            name="main_camera",
            width=self.width,
            height=self.height,
            fovy=np.deg2rad(35),
            near=0.1,
            far=100)

        camera_quat = euler_to_quat(roll=0, pitch=90, yaw=90, degrees=True).tolist()
        self.camera.set_pose(sapien.Pose([0, 0, 1], camera_quat))
        # self.camera.entity.set_pose(sapien.Pose(mat44))

    def _setup_light(self) -> None:
        """Setup robot light."""
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    def reset(self, initial_state: PhysicsState | None = None) -> tuple[PhysicsState, np.ndarray]:
        if initial_state is None:
            # Generate random position within reset area
            min_x, max_x, min_y, max_y = SAPIEN_DEFAULTS.reset_area
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            z = self._robot_radius + 0.005 # Ensure robot rests gently on the ground
            
            # Random initial yaw in degrees
            yaw_deg = float(np.random.uniform(-180, 180))

            initial_state = PhysicsState(
                position=[float(x), float(y), float(z)],
                velocity=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, yaw_deg],  # Euler degrees [roll, pitch, yaw]
                angular_velocity=[0.0, 0.0, 0.0]
            )

        self.robot.reset(initial_state)
        self.robot.update(initial_state.rotation[2])

        # Sync controller heading to the robot's initial yaw so the first
        # action uses the correct reference direction
        self.controller._reset()
        self.controller._heading_deg = initial_state.rotation[2]  # yaw in degrees

        for _ in range(10):
            self.scene.step()
            
        # Ensure viewer is updated after reset settling steps
        if self.viewer and not self.viewer.closed:
            self.scene.update_render()
            self.viewer.render()
            
        return self._get_state(), self._render_frame()

    def update_and_render(self, action: np.ndarray | list[float], dt: float | None = None) -> tuple[PhysicsState, np.ndarray]:
        """
        Action: [heading_deg, speed_0_255, duration_sec]
        """
        heading_deg, speed_255, duration_s = np.asarray(action, dtype=np.float32)

        # Set new commands
        self.controller.command(heading_deg, speed_255, duration_s)
        
        # Get physics component (for actors, this is PhysxRigidDynamicComponent)
        physics_comp = self.shell_link.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) if hasattr(self.shell_link, 'find_component_by_type') else None

        n_steps = 0
        while self.controller.phase !=  Phase.IDLE:
            # Get linear velocity
            if physics_comp:
                linear_vel = physics_comp.get_linear_velocity()
            else:
                linear_vel = self.shell_link.get_linear_velocity()
            v_linear = linear_vel[:2]

            #fx, fy = self.controller.get_forces(self.sim_dt, v_linear[0], v_linear[1])
            vx, vy = self.controller.get_velocity(self.sim_dt, v_linear[0], v_linear[1])

            # Apply force
            #self.robot.apply_force(fx, fy, self.sim_dt)
            self.robot.set_root_linear_velocity(vx=vx, vy=vy, vz=0)

            self.scene.step()
            
            # Update internal robot representation using the current commanded heading
            heading_deg_current = self.controller.heading_deg
            self.robot.update(heading_deg_current)

            # Update viewer if active
            # Only render every N steps to maintain performance
            if self.viewer and not self.viewer.closed and (n_steps % self.render_stride == 0):
                self.scene.update_render()
                self.viewer.render()
            n_steps += 1

        # Explicitly zero velocity when action is complete so physics doesn't drift
        self.robot.set_root_linear_velocity(vx=0, vy=0, vz=0)
        self.robot.set_root_angular_velocity(wx=0, wy=0, wz=0)
        self.scene.step()

        return self._get_state(), self._render_frame()

    def _get_state(self) -> PhysicsState:
        return self.robot.get_state()

    def _render_frame(self) -> np.ndarray:
        # Update scene rendering
        self.scene.update_render()
        
        # Take picture and get the rendered image
        self.camera.take_picture()
        
        # Get color image from camera (SAPIEN returns (H, W, 4) float32 in [0, 1])
        rgba = self.camera.get_picture('Color')
        
        # Convert to RGB uint8 [0, 255]
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return rgb

    def get_resolution(self) -> tuple[int, int, int]:
        """Get the resolution of the rendered image.

        Returns:
            tuple[int, int, int]: (height, width, channels).
        """
        return self.height, self.width, self.channels

    def get_ball_2d_position(self) -> tuple[int, int] | None:
        """Get the 2D pixel coordinates of the ball in the current rendered image."""
        if self.shell_link is None or self.camera is None:
            return None

        # Get 3D position of the ball
        ball_pos = self.shell_link.get_pose().p
        return self.project_position_to_2d(tuple(ball_pos))

    def get_reachable_bounds(self) -> tuple[float, float, float, float]:
        """Get the logical physical bounds where the ball can reach."""
        # Use the configured reset area as the reachable bounds for the goal
        min_x, max_x, min_y, max_y = SAPIEN_DEFAULTS.reset_area
        return float(min_x), float(min_y), float(max_x), float(max_y)

    def project_position_to_2d(self, position_3d: tuple[float, float, float]) -> tuple[int, int] | None:
        """Project a 3D physical position to 2D image coordinates."""
        if self.camera is None:
            return None

        # Project to 2D using camera
        cam_model = self.camera.get_intrinsic_matrix()

        # In Sapien, `get_extrinsic_matrix()` transforms world to camera: T_world_to_cam
        # This provides the transformation matrix from world-to-camera directly
        t_world_to_cam = self.camera.get_extrinsic_matrix()
        
        # Transform point to camera coordinates
        p_world = np.array(position_3d)
        p_world_h = np.append(p_world, 1.0)
        p_cam_h = t_world_to_cam @ p_world_h
        
        p_cam = p_cam_h[:3]

        # Ensure point is in front of camera
        if p_cam[2] <= 0:
            return None

        # Project to 2D
        p_img = cam_model @ p_cam
        x = int(p_img[0] / p_img[2])
        y = int(p_img[1] / p_img[2])

        # Check bounds
        if 0 <= x < self.width and 0 <= y < self.height:
            return x, y
        return None

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene with given configuration.

        Args:
            scene_config: Backend-specific scene configuration dictionary.
        """
        # Setup World (Ground + Robot) using Factory
        # Moved before viewer creation to match working example
        self.robot, self.shell_link, self.sled_link = self._setup_world()

        if "lighting" in scene_config:
            lighting = scene_config["lighting"]
            if "ambient_light" in lighting:
                self.scene.set_ambient_light(lighting["ambient_light"])
            if "directional_light" in lighting:
                light_dir, light_color = lighting["directional_light"]
                self.scene.add_directional_light(light_dir, light_color)
            if "point_light" in lighting:
                light_pos, light_color = lighting["point_light"]
                self.scene.add_point_light(light_pos, light_color)

        if "camera" in scene_config:
            camera_cfg = scene_config["camera"]
            if "pose" in camera_cfg:
                pos, quat = camera_cfg["pose"]
                self.camera.set_pose(sapien.Pose(pos, quat))
    
    def close(self) -> None:
        """Clean up resources and shutdown the SAPIEN engine."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.scene = None

