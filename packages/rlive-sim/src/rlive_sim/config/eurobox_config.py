"""Euro Box environment configuration.

Reads euro box floor material parameters from environment variables with
RLIVE_SIM_EUROBOX_ prefix.  These defaults are used by both GLBWorldStrategy
(real euro_box.glb mesh collision) and GLBFlatWorldStrategy (flat ground plane
used for calibration).

PhysX computes the contact friction as the geometric mean of both surfaces:
    effective_friction = sqrt(bolt_friction × eurobox_friction)

Set both surfaces to the same value to get predictable contact behaviour.
"""

import os
from dataclasses import dataclass


# ============================================================================
# FLOOR FRICTION CONFIGURATION
# ============================================================================

STATIC_FRICTION: float = float(os.getenv("RLIVE_SIM_EUROBOX_STATIC_FRICTION", "1.0"))
"""
Static friction of the euro box floor (resistance to starting motion).

default: 0.8
"""

DYNAMIC_FRICTION: float = float(os.getenv("RLIVE_SIM_EUROBOX_DYNAMIC_FRICTION", "0.1"))
"""
Dynamic friction of the euro box floor (resistance while moving).

default: 0.6
"""

RESTITUTION: float = float(os.getenv("RLIVE_SIM_EUROBOX_RESTITUTION", "0.0"))
"""
Restitution (bounciness) of the euro box floor surface.

0.0 = no bounce (soft rubber mat)

default: 0.0
"""

# ============================================================================
# DEFAULTS DATACLASS
# ============================================================================

@dataclass
class EuroBoxDefaults:
    """Physical material properties for the euro box floor.

    Applied to:
    - The euro_box.glb mesh collision (``robot_type="glb"``)
    - The synthetic flat ground plane (``robot_type="glb_flat"`` and
      ``robot_type="sapien"``)

    Attributes:
        static_friction:  Friction before the bolt starts moving.
        dynamic_friction: Friction while the bolt is rolling.
        restitution:      Bounciness of the floor (0 = no bounce).
    """
    static_friction:  float = STATIC_FRICTION
    dynamic_friction: float = DYNAMIC_FRICTION
    restitution:      float = RESTITUTION


EUROBOX_DEFAULTS = EuroBoxDefaults()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "STATIC_FRICTION",
    "DYNAMIC_FRICTION",
    "RESTITUTION",
    "EuroBoxDefaults",
    "EUROBOX_DEFAULTS",
]

