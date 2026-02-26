import cv2
import numpy as np
import math

def decode_fourcc(fourcc):
    return "".join([
        chr((int(fourcc) >> 8 * i) & 0xFF)
        for i in range(4)
    ])

def try_set_max_resolution(cap):
    test_resolutions = [
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1600, 1200),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
    ]

    best_w, best_h = 0, 0

    for w, h in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w * actual_h > best_w * best_h:
            best_w, best_h = actual_w, actual_h

    return best_w, best_h


def find_cameras_with_full_preview(max_index=10):
    print("Scanning cameras...\n")

    frames = []
    camera_results = []

    for index in range(max_index):
        print(f"--- Checking index {index} ---")

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("❌ Not available\n")
            continue

        ret, frame = cap.read()
        if not ret:
            print("⚠ Opened but no frame\n")
            cap.release()
            continue

        # Force highest possible resolution
        width, height = try_set_max_resolution(cap)

        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        backend = cap.getBackendName()

        print("✅ Camera found")
        print(f"Resolution : {width} x {height}")
        print(f"FPS        : {fps}")
        print(f"FOURCC     : {fourcc}")
        print(f"Backend    : {backend}\n")

        camera_results.append({
            "index": index,
            "resolution": f"{width} x {height}",
            "fps": fps,
            "fourcc": fourcc,
            "backend": backend
        })

        # Grab updated frame at max resolution
        ret, frame = cap.read()
        if not ret:
            cap.release()
            continue

        # Overlay all info onto image
        info_lines = [
            f"Index: {index}",
            f"Res: {width} x {height}",
            f"FPS: {fps:.2f}",
            f"FOURCC: {fourcc}",
            f"Backend: {backend}"
        ]

        y = 30
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            y += 30

        frames.append(frame)
        cap.release()

    if not frames:
        print("No cameras found.")
        return

    # Resize frames for grid
    target_width = 500
    resized_frames = []

    for frame in frames:
        h, w = frame.shape[:2]
        scale = target_width / w
        resized = cv2.resize(frame, (target_width, int(h * scale)))
        resized_frames.append(resized)

    cols = math.ceil(math.sqrt(len(resized_frames)))
    rows = math.ceil(len(resized_frames) / cols)

    max_height = max(img.shape[0] for img in resized_frames)

    padded_frames = []
    for img in resized_frames:
        h, w = img.shape[:2]
        if h < max_height:
            pad = np.zeros((max_height - h, w, 3), dtype=np.uint8)
            img = np.vstack((img, pad))
        padded_frames.append(img)

    grid_rows = []
    for r in range(rows):
        row_imgs = padded_frames[r*cols:(r+1)*cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(padded_frames[0]))
        grid_rows.append(np.hstack(row_imgs))

    final_image = np.vstack(grid_rows)

    print("====== RESULT ======\n")
    for cam in camera_results:
        print(f"--- Checking index {cam['index']} ---")
        print("✅ Camera found")
        print(f"Resolution : {cam['resolution']}")
        print(f"FPS        : {cam['fps']}")
        print(f"FOURCC     : {cam['fourcc']}")
        print(f"Backend    : {cam['backend']}\n")

    print("Showing combined preview window...")
    cv2.imshow("All Detected Cameras", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_cameras_with_full_preview(max_index=10)