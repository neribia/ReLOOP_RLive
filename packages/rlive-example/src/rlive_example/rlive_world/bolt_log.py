import csv
import math
import time
import numpy as np
from pathlib import Path

from rlive_common.utils import get_logger

from rlive_world.bolt.sphero_bolt_plus import SpheroBoltPlus

logger = get_logger(__name__)

# Gravity constant as defined in your API docs
G_TO_MS2 = 9.80665

def speed_ms(vel_dict):
    """Calculate speed magnitude from velocity components and convert cm/s to m/s."""
    # math.hypot(x, y) returns sqrt(x^2 + y^2)
    speed_cms = math.hypot(vel_dict["x"], vel_dict["y"])
    return speed_cms / 100.0  # Convert cm to m


def collect_sensor_data(robot, t, cmd, phase=None):
    """Collect and format sensor data for logging.
    Acceleration: Provides motion acceleration data along a given axis measured by the Accelerometer, in g's, where g = 9.80665 m/s^2.
        ``get_acceleration()['x']`` is the left-to-right acceleration, from -8 to 8 g's.
        ``get_acceleration()['y']`` is the forward-to-back acceleration, from of -8 to 8 g's.
        ``get_acceleration()['z']`` is the upward-to-downward acceleration, from -8 to 8 g's.

    Args:
        robot (SpheroBoltPlus): Connected robot instance.
        t (float): Current timestamp in seconds.
        cmd (int): Speed command (0-255).
        phase (str, optional): Current phase name ("idle", "steady", etc.).

    Returns:
        list: Formatted logging row.
              Units: time(s), cmd(int), vel(m/s), dist(m), accel(g)
    """
    # 1. Fetch raw data
    vel_raw = robot.api.get_velocity()
    dist_raw = robot.api.get_distance()
    accel_raw = robot.api.get_acceleration()

    # Phase logic
    if phase and phase != "steady":
        cmd = 0

    # 2. Process Velocity (Convert cm/s -> m/s)
    if vel_raw is not None:
        vx_m = round(vel_raw["x"] / 100.0, 4)
        vy_m = round(vel_raw["y"] / 100.0, 4)
        speed_m = round(math.hypot(vx_m, vy_m), 4)
    else:
        # If the API returned None, we log NaN
        vx_m = vy_m = speed_m = np.nan

    # 3. Process Distance (Convert cm -> m)
    dist_m = round(dist_raw / 100.0, 4) if dist_raw is not None else np.nan

    # 4. Process Acceleration (Convert g's -> m/s2)
    if accel_raw is not None:
        # Multiply each axis by the gravity constant before rounding
        ax = round(accel_raw["x"] * G_TO_MS2, 4)
        ay = round(accel_raw["y"] * G_TO_MS2, 4)
        az = round(accel_raw["z"] * G_TO_MS2, 4)
    else:
        ax = ay = az = np.nan

    # 5. Build the final list
    logging_list = [
        round(t, 4),
        cmd,
        vx_m, vy_m, speed_m,
        dist_m,
        ax, ay, az
    ]

    return logging_list


def phase_for_time(t: float, idle_s: float, steady_s: float) -> str:
    """Return the current experiment phase for a timestamp."""
    if t < idle_s:
        return "idle"
    if t < idle_s + steady_s:
        return "steady"
    return "ramp-down"


def log_roll_with_ramps(robot, cmd, idle_s, steady_s, ramp_down_s, csv_path, sample_hz=60):
    """Log sensor data capturing idle, steady-state, and ramp-down phases.

    Timeline:
    - Phase 1 (idle): Log for idle_s seconds BEFORE issuing speed command (0 ≤ t < idle_s)
    - Phase 2 (steady): Issue speed command, log for steady_s seconds (idle_s ≤ t < idle_s+steady_s)
    - Phase 3 (ramp-down): Set speed to 0, log for ramp_down_s seconds (idle_s+steady_s ≤ t < total)

    Args:
        robot (SpheroBoltPlus): Connected Sphero BOLT+ robot instance.
        cmd (int): Speed command (0-255) to apply during steady phase.
        idle_s (float): Duration to log BEFORE issuing speed command (seconds).
        steady_s (float): Duration to hold speed command (seconds).
        ramp_down_s (float): Duration to log after setting speed to 0 (seconds).
        csv_path (str): Path to output CSV file.
        sample_hz (int): Sampling frequency in Hz. Default: 60 Hz.

    Returns:
        None
    """
    dt = 1.0 / sample_hz
    total_duration = idle_s + steady_s + ramp_down_s

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "cmd", "vx_ms", "vy_ms", "speed_ms", "travel_distance_m", "accel_x_ms2", "accel_y_ms2", "accel_z_ms2"])

        logger.info(f"Starting experiment")
        logger.info(f"  Sampling frequency: {sample_hz} ({dt:.3f}s)")
        logger.info(f"  Phase 1 (idle): t=0.0-{idle_s}s (logging BEFORE speed command)")
        logger.info(f"  Phase 2 (steady): t={idle_s}-{idle_s + steady_s}s (cmd={cmd})")
        logger.info(f"  Phase 3 (ramp-down): t={idle_s + steady_s}-{total_duration}s (cmd=0)")

        speed_sent = False
        stop_sent = False
        t0 = time.perf_counter()
        next_sample_time = t0

        while True:
            now = time.perf_counter()
            if now < next_sample_time:
                time.sleep(next_sample_time - now)
                #logger.info(f"Sleeping for: {next_sample_time - now:.3f}s until next sample")
                continue

            t = time.perf_counter() - t0

            if t >= total_duration:
                break

            phase = phase_for_time(t, idle_s, steady_s)

            # Issue commands exactly when the phase changes.
            if phase == "steady" and not speed_sent:
                robot.set_speed(cmd)
                logger.info(f"[t={t:.3f}s] Set speed to {cmd} (steady-state starts)")
                speed_sent = True

            if phase == "ramp-down" and not stop_sent:
                robot.stop_roll()
                logger.info(f"[t={t:.3f}s] Set speed to 0 (ramp-down starts)")
                stop_sent = True

            logging_list = collect_sensor_data(robot, t, cmd, phase)
            w.writerow(logging_list)

            next_sample_time += dt

        # Capture one extra sample right after the nominal end time.
        t_final = time.perf_counter() - t0
        final_phase = phase_for_time(t_final, idle_s, steady_s)
        final_row = collect_sensor_data(robot, t_final, cmd, final_phase)
        w.writerow(final_row)

        logger.info(f"Experiment complete at t={t_final:.3f}s (including final post-duration sample)")


if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    # Example connection with auto-connect in context manager
    toy_name = "BP-D217"
    speed = 150  # Speed command (0-255)
    duration_s = 1.0



    # Log: idle (0.5s) → speed command → steady (1.0s) → speed 0 → ramp-down (0.5s)
    with SpheroBoltPlus(bolt_name=toy_name, timeout=3) as robot:
        log_roll_with_ramps(
            robot,
            cmd=speed,
            idle_s=0.5,                 # Log for 0.5s BEFORE issuing speed command
            steady_s=duration_s,        # Hold speed command for 1.0s
            ramp_down_s=5.0,            # Log deceleration for 0.5s
            csv_path= str(data_dir / f"exp_speed_1s_{speed:03}.csv"),
        )
