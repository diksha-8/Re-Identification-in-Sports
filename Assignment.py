import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv11 model 
model = YOLO("best.pt")  

# Initialize DeepSORT with appearance-based embedder
tracker = DeepSort(
    max_age=60,
    n_init=2,
    max_cosine_distance=0.4,
    nn_budget=100,
    embedder="mobilenet",  # Use appearance features
    half=True,
)

# Load video 
cap = cv2.VideoCapture("15sec_input_720p.mp4")
assert cap.isOpened(), "Error opening video file"

# Standardize frame size to 720p HD 
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Output video writer 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_reid.mp4", fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to standard resolution
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Detect with YOLO
    results = model(frame)[0]

    detections = []
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result
        if conf < 0.2:  # Lower threshold to catch all players
            continue
        if int(cls) == 0:  # class 0 = player
            bbox = [x1.item(), y1.item(), x2.item() - x1.item(), y2.item() - y1.item()]
            detections.append((bbox, conf.item(), 'player'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Tracking Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
