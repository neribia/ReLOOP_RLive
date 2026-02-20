"""Tracking Demo Script for visual inspection of ball detection.

This script allows you to test the ball localization using either a webcam
or a video file. It displays the detection results in real-time with annotations.

Usage:
    # Using webcam (default, camera index 0)
    python tracking_demo.py

    # Using specific webcam
    python tracking_demo.py --source 1

    # Using video file
    python tracking_demo.py --source path/to/video.mp4

    # With custom resolution
    python tracking_demo.py --width 1280 --height 720

Controls:
    - 'q' or ESC: Quit
    - 'g': Set goal position to current mouse position
    - 'm': Toggle detection method
    - 'd': Toggle debug view
    - 's': Save current frame
    - 'r': Reset goal position
"""

import sys
from pathlib import Path
import time

import cv2 as cv
import numpy as np

# Add the packages to path for direct script execution
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "packages" / "rlive-env" / "src"))
sys.path.insert(0, str(SCRIPT_DIR / "packages" / "rlive-common" / "src"))

from rlive_env.localisation import BallLocalisator, BallLocation
from rlive_env.localisation.processors import *
from rlive_env.localisation.extractors import *


class KeyHandler:
    """Handles keyboard input and key events."""

    # Key codes for different platforms
    KEY_Q = ord('q')
    KEY_ESC = 27
    KEY_D = ord('d')
    KEY_S = ord('s')
    KEY_R = ord('r')
    KEY_G = ord('g')
    KEY_SPACE = ord(' ')
    KEY_M = ord('m')
    KEY_PLUS = ord('+')
    KEY_MINUS = ord('-')

    # Number keys (top row)
    KEY_1 = ord('1')
    KEY_2 = ord('2')

    # Arrow keys - different codes for different platforms
    KEY_LEFT = 65361  # Left arrow
    KEY_RIGHT = 65363  # Right arrow
    KEY_UP = 65362  # Up arrow
    KEY_DOWN = 65364  # Down arrow

    # Alternative arrow key codes (some systems)
    KEY_LEFT_ALT = 81  # Left
    KEY_RIGHT_ALT = 83  # Right
    KEY_UP_ALT = 82  # Up
    KEY_DOWN_ALT = 84  # Down

    def __init__(self):
        """Initialize key handler."""
        pass

    @staticmethod
    def is_quit(key: int) -> bool:
        """Check if quit key was pressed."""
        return key == KeyHandler.KEY_Q or key == KeyHandler.KEY_ESC

    @staticmethod
    def is_debug_toggle(key: int) -> bool:
        """Check if debug toggle key was pressed."""
        return key == KeyHandler.KEY_D

    @staticmethod
    def is_save(key: int) -> bool:
        """Check if save key was pressed."""
        return key == KeyHandler.KEY_S

    @staticmethod
    def is_reset_goal(key: int) -> bool:
        """Check if reset goal key was pressed."""
        return key == KeyHandler.KEY_R

    @staticmethod
    def is_set_goal(key: int) -> bool:
        """Check if set goal key was pressed."""
        return key == KeyHandler.KEY_G

    @staticmethod
    def is_manual_step(key: int) -> bool:
        """Check if manual frame step key was pressed (space)."""
        return key == KeyHandler.KEY_SPACE

    @staticmethod
    def is_increase_skip(key: int) -> bool:
        """Check if increase skip frames key was pressed (+ or =)."""
        return key == KeyHandler.KEY_PLUS or key == ord('=')

    @staticmethod
    def is_decrease_skip(key: int) -> bool:
        """Check if decrease skip frames key was pressed (-)."""
        return key == KeyHandler.KEY_MINUS

    @staticmethod
    def is_toggle_manual_mode(key: int) -> bool:
        """Check if toggle manual mode key was pressed (m)."""
        return key == KeyHandler.KEY_M

    @staticmethod
    def is_previous_stage(key: int) -> bool:
        """Check if previous stage key was pressed (left arrow or up arrow)."""
        return key in (
            KeyHandler.KEY_LEFT,
            KeyHandler.KEY_LEFT_ALT,
            KeyHandler.KEY_UP,
            KeyHandler.KEY_UP_ALT,
            KeyHandler.KEY_1)

    @staticmethod
    def is_next_stage(key: int) -> bool:
        """Check if next stage key was pressed (right arrow or down arrow)."""
        return key in (
            KeyHandler.KEY_RIGHT,
            KeyHandler.KEY_RIGHT_ALT,
            KeyHandler.KEY_DOWN,
            KeyHandler.KEY_DOWN_ALT,
            KeyHandler.KEY_2
        )


class VideoSource:
    """Video source wrapper for webcam or video file input.

    Provides a unified interface for capturing frames from either
    a webcam (by index) or a video file (by path).
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
    ) -> None:
        """Initialize the video source.

        Args:
            source: Camera index (int) or video file path (str).
            width: Desired frame width (only applies to webcam).
            height: Desired frame height (only applies to webcam).
        """
        self.source = source
        self.width = width
        self.height = height
        self._cap: cv.VideoCapture | None = None
        self._is_file = isinstance(source, str)
        # Target frames per second to enforce on the consumer side.
        # We keep timing state here so the VideoSource can control frame pacing
        # uniformly for any consumer of frames.
        self.target_fps = int(target_fps) if target_fps and target_fps > 0 else 30
        self.prev_time = time.time()

    def create(self) -> cv.VideoCapture:
        """Create and configure the VideoCapture object.

        Returns:
            Configured cv2.VideoCapture instance.

        Raises:
            RuntimeError: If the video source cannot be opened.
        """
        if isinstance(self.source, str):
            # Video file
            if not Path(self.source).exists():
                raise RuntimeError(f"Video file not found: {self.source}")
            self._cap = cv.VideoCapture(self.source)
        else:
            # Webcam
            self._cap = cv.VideoCapture(self.source, cv.CAP_DSHOW)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        # Set properties for webcam (best-effort; drivers may ignore some settings)
        if not self._is_file:
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
            # Attempt to set capture FPS to the requested target
            try:
                self._cap.set(cv.CAP_PROP_FPS, float(self.target_fps))
            except Exception:
                # Some OpenCV backends may raise or ignore this; ignore failures
                pass

        return self._cap

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_file(self) -> bool:
        """Check if source is a video file."""
        return self._is_file

    def get_info(self) -> dict:
        """Get information about the video source."""
        if self._cap is None:
            return {}

        return {
            "width": int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv.CAP_PROP_FPS),
            "frame_count": int(self._cap.get(cv.CAP_PROP_FRAME_COUNT)) if self._is_file else -1,
        }

    def wait_for_frame(self) -> None:
        """Block as needed to maintain the configured target_fps.

        Call this once per consumer loop iteration after processing/displaying
        a frame. It sleeps the minimum required time to reach the target FPS
        based on the last recorded frame time.
        """
        if not self.target_fps or self.target_fps <= 0:
            return
        current_time = time.time()
        target_interval = 1.0 / float(self.target_fps)
        wait_time = target_interval - (current_time - self.prev_time)
        if wait_time > 0:
            time.sleep(wait_time)
        # update prev_time to current time after sleeping
        self.prev_time = time.time()


class TrackingDemo:
    """Interactive demo for testing ball tracking/localization."""

    def __init__(
        self,
        source: int | str,
        width: int,
        height: int,
        target_fps: int,
        goal_radius: int = 50,
        show_debug_on_start: bool = True,
        custom_pipeline: ImagePipeline | None = None,
        custom_extractor = None,
        scale_factor: float = 1.0,
        manual_mode: bool = False,
        skip_frames: int = 0,
    ) -> None:
        """Initialize the tracking demo.

        Args:
            source: Camera index (int) or video file path (str)
            width: Frame width
            height: Frame height
            target_fps: Target frames per second
            goal_radius: Radius around goal for success detection
            show_debug_on_start: Show debug view on startup
            custom_pipeline: Optional custom ImagePipeline to use
            custom_extractor: Optional custom extractor instance to use
            scale_factor: Scale factor for display windows (1.0 = original size, 0.5 = half size, etc.)
            manual_mode: Enable manual frame stepping (press SPACE to go to next frame)
            skip_frames: Number of frames to skip between detections (0 = no skip, process every frame)
        """
        # Initialize video source
        self.video_source = VideoSource(source, width, height, target_fps=target_fps)
        self.localiser = BallLocalisator()

        # Use custom pipeline if provided
        if custom_pipeline is not None:
            self.localiser.pipeline = custom_pipeline

        # Use custom extractor if provided
        if custom_extractor is not None:
            self.localiser.extractor = custom_extractor

        self.goal_position: tuple[int, int] | None = None
        self.goal_radius = goal_radius
        self.mouse_position = (0, 0)

        self.frame_count = 0
        self.save_count = 0
        self.show_debug = show_debug_on_start
        self.current_stage = 0
        self.debug_stages: list[np.ndarray] = []

        # Window scale factor
        self.scale_factor = scale_factor

        # Manual mode and frame skipping
        self.manual_mode = manual_mode
        self.skip_frames = skip_frames
        self.frame_step_requested = False
        self.frames_skipped = 0

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events."""
        self.mouse_position = (x, y)

        if event == cv.EVENT_LBUTTONDOWN:
            self.goal_position = (int(x / self.scale_factor), int(y / self.scale_factor))
            print(f"Goal set to: ({x}, {y})")

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame based on scale factor.

        Args:
            frame: Input frame to resize

        Returns:
            Resized frame or original if scale_factor is 1.0
        """
        if self.scale_factor == 1.0:
            return frame

        height, width = frame.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        return cv.resize(frame, (new_width, new_height))

    def _draw_overlay(self, frame: np.ndarray, ball: BallLocation | None) -> np.ndarray:
        """Draw detection overlay on frame."""
        # Use localiser's annotate method
        annotated_frame = frame.copy()
        annotated_frame = self._draw_goal(annotated_frame, self.goal_position, self.goal_radius)

        annotated = self.localiser.annotate_image(
            annotated_frame,
            ball_location=ball,
            goal_position=self.goal_position,
            goal_radius=self.goal_radius,
        )

        # Draw info text
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Ball: {ball.as_tuple() if ball else 'Not detected'}",
            f"Goal: {self.goal_position if self.goal_position else 'Click to set'}",
        ]

        if self.debug_stages:
            info_lines.append(f"Debug Stage: {self.current_stage + 1}/{len(self.debug_stages)}")

        # Add manual mode and skip frames info
        if self.manual_mode:
            info_lines.append(f"Mode: MANUAL (Press SPACE for next frame)")
        else:
            info_lines.append(f"Mode: AUTO")

        if self.skip_frames > 0:
            info_lines.append(f"Skip Frames: {self.skip_frames} ({self.frames_skipped} skipped)")

        if ball and self.goal_position:
            distance = ball.distance_to(self.goal_position)
            info_lines.append(f"Distance: {distance:.1f}px")
            if ball.is_within_radius(self.goal_position, self.goal_radius):
                info_lines.append("STATUS: GOAL REACHED!")
                cv.rectangle(annotated, (0, 0), (250, 30), (0, 255, 0), -1)

        y_offset = 20
        for line in info_lines:
            cv.putText(annotated, line, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        # Draw controls help at bottom
        help_text = "Q:Quit | Click:Goal | D:Debug | M:Manual | +:SkipPlus | -:SkipMinus | SPACE:NextFrame | S:Save | R:Reset"
        h = annotated.shape[0]
        cv.putText(annotated, help_text, (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return annotated

    def _handle_key_input(self, key: int, display_frame: np.ndarray, debug_window_name: str) -> bool:
        """Handle keyboard input and return True if should quit.

        Args:
            key: Key code from cv.waitKey()
            display_frame: Current display frame for saving
            debug_window_name: Name of debug window

        Returns:
            True if quit was requested, False otherwise
        """
        if KeyHandler.is_quit(key):
            return True
        elif KeyHandler.is_debug_toggle(key):
            self.show_debug = not self.show_debug
            print(f"Debug view: {'ON' if self.show_debug else 'OFF'}")
            if not self.show_debug:
                try:
                    cv.destroyWindow(debug_window_name)
                except cv.error:
                    pass
        elif KeyHandler.is_previous_stage(key):
            if self.debug_stages and self.current_stage > 0:
                self.current_stage -= 1
        elif KeyHandler.is_next_stage(key):
            if self.debug_stages and self.current_stage < len(self.debug_stages) - 1:
                self.current_stage += 1
        elif KeyHandler.is_save(key):
            filename = f"localisation_frame_{self.save_count:04d}.png"
            cv.imwrite(filename, display_frame)
            print(f"Saved: {filename}")
            self.save_count += 1
        elif KeyHandler.is_reset_goal(key):
            self.goal_position = None
            print("Goal reset")
        elif KeyHandler.is_set_goal(key):
            self.goal_position = self.mouse_position
            print(f"Goal set to mouse position: {self.goal_position}")
        elif KeyHandler.is_toggle_manual_mode(key):
            self.manual_mode = not self.manual_mode
            self.frame_step_requested = False
            print(f"Manual mode: {'ON (Press SPACE for next frame)' if self.manual_mode else 'OFF (Auto playback)'}")
        elif KeyHandler.is_manual_step(key):
            if self.manual_mode:
                self.frame_step_requested = True
                print("Manual step: Next frame requested")
        elif KeyHandler.is_increase_skip(key):
            self.skip_frames += 5
            self.frames_skipped = 0
            print(f"Skip frames set to: {self.skip_frames}")
        elif KeyHandler.is_decrease_skip(key):
            self.skip_frames = max(0, self.skip_frames - 5)
            self.frames_skipped = 0
            print(f"Skip frames set to: {self.skip_frames}")

        return False

    @staticmethod
    def _draw_goal(image, goal_position, goal_radius):
        annotated = image.copy()
        # Draw goal if provided
        if goal_position is not None:
            cv.circle(annotated, goal_position, goal_radius, (0, 255, 0), 2)
            cv.circle(annotated, goal_position, 5, (0, 255, 0), -1)
            return annotated

        return image


    def run(self) -> None:
        """Run the tracking demo main loop."""
        cap = self.video_source.create()
        info = self.video_source.get_info()

        # Print info; the VideoSource is responsible for pacing (target_fps)
        source_fps = info.get("fps", None)
        if source_fps and source_fps > 0:
            print(f"Video source info: {info} (source FPS: {source_fps:.1f})")
        else:
            print(f"Video source info: {info}")
        print(f"Target playback: {self.video_source.target_fps} FPS")

        print("\nControls:")
        print("  Click: Set goal position")
        print("  Q/ESC: Quit")
        print("  D: Toggle debug view")
        print("  M: Toggle manual mode (press SPACE to step through frames)")
        print("  +/-: Increase/decrease skip frames (for later use in actual implementation)")
        print("  Left/Right Arrow: Navigate debug stages")
        print("  S: Save current frame")
        print("  R: Reset goal\n")

        if self.manual_mode:
            print("Starting in MANUAL MODE - Press SPACE to advance frames")
        if self.skip_frames > 0:
            print(f"Skip frames is set to: {self.skip_frames}")

        window_name = "Ball Localisation Demo"
        debug_window_name = "Debug View"
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self._mouse_callback)

        try:
            while True:
                if self.video_source.is_file:
                    self.video_source.wait_for_frame()
                ret, frame = cap.read()

                if not ret:
                    if self.video_source.is_file:
                        cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Loop video file
                        continue
                    else:
                        print("Failed to capture frame")
                        break

                self.frame_count += 1

                # Handle frame skipping
                if self.skip_frames > 0:
                    if self.frames_skipped < self.skip_frames:
                        self.frames_skipped += 1
                        # Display the frame but skip processing for now
                        display_frame = self._draw_overlay(frame, None)
                        cv.imshow(window_name, self._resize_frame(display_frame))

                        key = cv.waitKey(1) & 0xFF
                        if self._handle_key_input(key, display_frame, debug_window_name):
                            break
                        continue
                    else:
                        self.frames_skipped = 0

                # Execute pipeline with debug=True to capture intermediate stages
                self.localiser.pipeline.execute(frame, debug=True)
                self.debug_stages = self.localiser.pipeline.get_intermediate_results()

                # Get ball position using the debug-executed pipeline result
                result = self.localiser.get_position_with_debug(frame)
                ball = result.location if result else None

                display_frame = self._draw_overlay(frame, ball)
                cv.imshow(window_name, self._resize_frame(display_frame))

                # Display current debug stage
                if self.show_debug and self.debug_stages:
                    debug_img = self.debug_stages[self.current_stage]
                    if len(debug_img.shape) == 2:
                        debug_img = cv.cvtColor(debug_img, cv.COLOR_GRAY2BGR)

                    # Add stage name label to debug image
                    num_pipeline_steps = len(self.localiser.pipeline.steps)
                    stage_label = f"Stage {self.current_stage}/{len(self.debug_stages) - 1}: "
                    if self.current_stage == 0:
                        stage_label += "Original"
                    elif self.current_stage <= num_pipeline_steps:
                        step = self.localiser.pipeline.steps[self.current_stage - 1]
                        stage_label += getattr(step, "name", type(step).__name__)
                    else:
                        # Extractor debug image (appended after pipeline steps)
                        stage_label += f"Extractor: {self.localiser.extractor.name}"

                    cv.rectangle(debug_img, (0, 0), (len(stage_label) * 11, 28), (0, 0, 0), -1)
                    cv.putText(debug_img, stage_label, (5, 20),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv.imshow(debug_window_name, self._resize_frame(debug_img))
                elif not self.show_debug:
                    try:
                        if cv.getWindowProperty(debug_window_name, cv.WND_PROP_VISIBLE) >= 1:
                            cv.destroyWindow(debug_window_name)
                    except:
                        pass

                # Handle keyboard input and frame stepping
                if self.manual_mode:
                    # In manual mode, wait for key press (0 timeout = wait indefinitely)
                    key = cv.waitKey(0) & 0xFF
                    if self._handle_key_input(key, display_frame, debug_window_name):
                        break
                    # Check if SPACE was pressed to advance
                    if not self.frame_step_requested:
                        # If SPACE wasn't pressed, wait again
                        continue
                    self.frame_step_requested = False
                else:
                    # In auto mode, check for key presses with short timeout
                    key = cv.waitKey(1) & 0xFF
                    if self._handle_key_input(key, display_frame, debug_window_name):
                        break


        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.video_source.release()
            cv.destroyAllWindows()


def main():
    """Main entry point for the tracking demo."""

    # ========== CONFIGURATION ==========
    # Video source settings
    VIDEO_SOURCE = 1# "demo_video.mp4"  # Use 0 for webcam, or path to video file
    FRAME_WIDTH = 640 # Frame width for webcam
    FRAME_HEIGHT = 480 # Frame height for webcam
    TARGET_FPS = 30  # Target frames per second

    # Goal settings
    GOAL_RADIUS = 50  # Radius around goal for success detection

    # Debug settings
    SHOW_DEBUG_ON_START = True  # Show debug view on startup

    # Display settings
    SCALE_FACTOR = 2.0  # Scale factor for display windows (1.0 = original, 0.5 = half size, 2.0 = double size)

    # Manual mode and frame skipping
    MANUAL_MODE = False  # Set to True to enable manual frame stepping (press SPACE for next frame)
    SKIP_FRAMES = 0  # Number of frames to skip between detections (0 = no skip, process every frame)
                      # Example: 5 will skip 5 frames, process the 6th frame, then skip 5 more, etc.

    # Custom pipeline processors
    custom_pipeline = ImagePipeline([
        HSVProcessor(
            lower_hue=90, upper_hue=130,  # blue hue range
            lower_sat=50, upper_sat=255,  # require some color (not gray)
            lower_val=50, upper_val=255,  # require some brightness (not black)
            apply_mask=True,
        ),
        DilationProcessor(kernel_size=(7, 7), iterations=3),
    ])

    # Custom extractor configuration - loads defaults from config module
    # Override specific parameters as needed:
    custom_extractor = ContourExtractor(min_contour_area=50)
    # Or use defaults:
    # custom_extractor = ContourExtractor()
    # ====================================


    demo = TrackingDemo(
        source=VIDEO_SOURCE,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        target_fps=TARGET_FPS,
        goal_radius=GOAL_RADIUS,
        show_debug_on_start=SHOW_DEBUG_ON_START,
        custom_pipeline=custom_pipeline,
        custom_extractor=custom_extractor,
        scale_factor=SCALE_FACTOR,
        manual_mode=MANUAL_MODE,
        skip_frames=SKIP_FRAMES,
    )
    demo.run()


if __name__ == "__main__":
    main()
