"""Robot abstractions for SAPIEN integration."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

try:
    import sapien
except ImportError:
    sapien = None

from rlive_sim.engine.core.base_physics_engine import PhysicsState
from rlive_sim.utils.math_utils import euler_to_quat, quat_to_euler
from rlive_sim.config.bolt_config import BOLT_DEFAULTS, BoltDefaults


class SapianRobot(ABC):
    """Base class for robots in the integrated engine."""
    
    def __init__(self, scene: "sapien.Scene", config: Dict[str, Any]):
        self.scene = scene
        self.config = config
        self.articulation: Optional[sapien.Articulation] = None
        self.shell_link: Optional[sapien.Entity] = None
        
    @abstractmethod
    def load(self) -> None:
        """Load the robot into the SAPIEN scene."""
        pass

    @abstractmethod
    def reset(self, initial_state: Optional[PhysicsState] = None) -> None:
        """Reset the robot to its initial state."""
        pass

    @abstractmethod
    def apply_force(self, fx: float, fy: float, dt: float) -> None:
        """Apply control forces to the robot."""
        pass

    @abstractmethod
    def update(self, heading_deg: float = 0.0) -> None:
        """Update any kinematic or visual mechanisms (called each sub-step)."""
        pass

    def get_state(self) -> PhysicsState:
        """Get the current physics state."""
        if self.shell_link is None:
            raise RuntimeError("Robot not properly loaded.")
            
        pose = self.shell_link.pose
        
        # Get physics component to access velocity if it's an actor
        physics_comp = (self.shell_link.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) 
                        if hasattr(self.shell_link, 'find_component_by_type') else None)
        
        if physics_comp:
            linear_vel = physics_comp.get_linear_velocity()
            angular_vel = physics_comp.get_angular_velocity()
        else:
            linear_vel = self.shell_link.get_linear_velocity()
            angular_vel = self.shell_link.get_angular_velocity()
            
        return PhysicsState(
            position=pose.p.tolist(),
            velocity=linear_vel.tolist(),
            rotation=quat_to_euler(pose.q, degrees=True).tolist(),  # Convert SAPIEN quat → Euler deg
            angular_velocity=angular_vel.tolist()
        )


class SimpleSphereRobot(SapianRobot):
    """Programmatically built rigid sphere."""
    
    def load(self) -> None:
        builder = self.scene.create_articulation_builder()
        shell_link_builder = builder.create_link_builder()
        shell_link_builder.set_name("shell")
        
        bolt_cfg: BoltDefaults = self.config.get('bolt_config', BOLT_DEFAULTS)
        shell_radius = self.config.get('robot_radius', bolt_cfg.radius_m)
        mass_shell = self.config.get('robot_mass', bolt_cfg.mass_kg)

        shell_mat = self.scene.create_physical_material(
            static_friction=bolt_cfg.static_friction,
            dynamic_friction=bolt_cfg.dynamic_friction,
            restitution=bolt_cfg.restitution_coefficient,
        )
        
        shell_link_builder.add_sphere_collision(radius=shell_radius, material=shell_mat)
        shell_link_builder.add_sphere_visual(radius=shell_radius)
        
        inertia_shell = (2/3) * mass_shell * (shell_radius**2)
        shell_link_builder.set_mass_and_inertia(
            mass_shell,
            sapien.Pose(),
            [inertia_shell, inertia_shell, inertia_shell]
        )
        
        self.articulation = builder.build(fix_root_link=False)
        self.articulation.set_name("simple_sphere")
        self.shell_link = self.articulation.get_links()[0]
        self.reset()
        
    def reset(self, initial_state: Optional[PhysicsState] = None) -> None:
        if self.articulation is None:
            return
            
        shell_radius = self.config.get('robot_radius', 0.0365)
        self.articulation.set_pose(sapien.Pose([0, 0, shell_radius]))
        self.articulation.set_qpos(np.zeros(self.articulation.dof))
        self.articulation.set_qvel(np.zeros(self.articulation.dof))
        
        if initial_state:
            self.articulation.set_pose(sapien.Pose(initial_state.position))
            
    def apply_force(self, fx: float, fy: float, dt: float) -> None:
        self.shell_link.add_force_at_point([fx, fy, 0], self.shell_link.pose.p)

    def update(self, heading_deg: float = 0.0) -> None:
        pass


class KinematicSpheroRobot(SapianRobot):
    """
    A sphere shell for physics and an internal payload actor updated kinematically.
    """
    
    def __init__(self, scene: "sapien.Scene", config: Dict[str, Any], shell_path: str, robot_path: str):
        super().__init__(scene, config)
        self.shell_path = shell_path
        self.robot_path = robot_path
        self.internal_actor: Optional[sapien.Entity] = None
        # Default offset to correct the GLB model's inherent rotation around the Z axis
        self.yaw_offset_deg = -90.0
        
    def load(self) -> None:
        # 1. Build dynamic physics shell
        builder = self.scene.create_articulation_builder()
        shell_link_builder = builder.create_link_builder()
        shell_link_builder.set_name("shell")
        
        shell_link_builder.add_visual_from_file(self.shell_path)
            
        bolt_cfg: BoltDefaults = self.config.get('bolt_config', BOLT_DEFAULTS)
        shell_radius = self.config.get('robot_radius', bolt_cfg.radius_m)
        mass_robot = self.config.get('robot_mass', bolt_cfg.mass_kg)
        mass_shell = mass_robot * 0.1

        shell_mat = self.scene.create_physical_material(
            static_friction=bolt_cfg.static_friction,
            dynamic_friction=bolt_cfg.dynamic_friction,
            restitution=bolt_cfg.restitution_coefficient,
        )
        
        shell_link_builder.add_sphere_collision(radius=shell_radius, material=shell_mat, density=100.0)
        
        inertia_shell = (2/5) * mass_robot * (shell_radius**2)
        com_offset = sapien.Pose([0, 0, -shell_radius * 0.5])
        shell_link_builder.set_mass_and_inertia(
            mass_robot,
            com_offset,
            [inertia_shell, inertia_shell, inertia_shell]
        )
        
        self.articulation = builder.build(fix_root_link=False)
        self.articulation.set_name("sphero_kinematic_shell")
        self.shell_link = self.articulation.get_links()[0]

        linear_damping = self.config.get('linear_damping', 0.0)
        angular_damping = self.config.get('angular_damping', 0.0)

        self.shell_link.set_linear_damping(linear_damping)
        self.shell_link.set_angular_damping(angular_damping)
        
        # 2. Build kinematic internal visual robot
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_visual_from_file(self.robot_path)
            
        # Kinematic means it doesn't do physics, we update pose manually
        self.internal_actor = actor_builder.build_kinematic(name="sphero_internal_robot")
        
        self.reset()
        
    def reset(self, initial_state: Optional[PhysicsState] = None) -> None:
        if self.articulation is None:
            return
            
        shell_radius = self.config.get('robot_radius', 0.0365)
        self.articulation.set_pose(sapien.Pose([0, 0, shell_radius + 0.005]))
        self.articulation.set_qpos(np.zeros(self.articulation.dof))
        self.articulation.set_qvel(np.zeros(self.articulation.dof))
        
        if initial_state:
            # Convert Euler degrees → quaternion for SAPIEN Pose
            q = euler_to_quat(*initial_state.rotation, degrees=True).reshape(4)
            self.articulation.set_pose(sapien.Pose(initial_state.position, q))

        self._sync_internal_pose(0.0)
        
    def apply_force(self, fx: float, fy: float, dt: float) -> None:
        if self.shell_link is None:
            return

        shell_radius = self.config.get('robot_radius', 0.0365)

        # We want to apply the force at the inside bottom of the shell
        # Transform the local bottom point (0, 0, -radius) to world coordinates
        pose = self.shell_link.pose

        # SAPIEN's add_force_at_point expects world coordinates for both the force and the point.
        # So we take the ball's center and subtract the radius on the Z-axis.
        contact_point = pose.p + np.array([0, 0, -shell_radius])

        # Apply the linear force to the bottom, which creates natural rolling torque
        self.shell_link.add_force_at_point([fx, fy, 0], contact_point)

    def set_root_linear_velocity(self, vx: float, vy: float, vz: float) -> None:
        if self.articulation is None:
            return
        self.articulation.set_root_linear_velocity(np.array([vx, vy, vz], dtype=np.float32))

    def set_root_angular_velocity(self, wx: float, wy: float, wz: float) -> None:
        if self.articulation is None:
            return
        self.articulation.set_root_angular_velocity(np.array([wx, wy, wz], dtype=np.float32))

    def update(self, heading_deg: float = 0.0) -> None:
        self._sync_internal_pose(heading_deg)
        
    def _sync_internal_pose(self, heading_deg: float) -> None:
        if self.internal_actor is None or self.shell_link is None:
            return
            
        # Get position of shell
        shell_pos = self.shell_link.pose.p
        
        # Internal actor stays upright with its own Z aligned to World Z, but yaw changes to intended heading
        # Apply the visual offset to align the model properly
        q_rot = euler_to_quat(0, 0, heading_deg + self.yaw_offset_deg, degrees=True)  # Yaw around Z
        self.internal_actor.set_pose(sapien.Pose(shell_pos, q_rot))
