import cv2
import time
import math

# Load video
cap = cv2.VideoCapture("car.mp4")

# Object detector (pretrained Haar or YOLO/others can be used)
car_cascade = cv2.CascadeClassifier("cars.xml")

# Parameters
pixels_per_meter = 8  # <-- You need to calibrate this for real-world scaling
prev_positions = {}
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    for (x, y, w, h) in cars:
        cx = int(x + w/2)
        cy = int(y + h/2)

        car_id = f"{x}-{y}"  # simple ID (can use tracking library instead)
        if car_id in prev_positions:
            (px, py) = prev_positions[car_id]

            # Distance in pixels
            dist_pixels = math.hypot(cx - px, cy - py)

            # Convert to meters
            dist_meters = dist_pixels / pixels_per_meter

            # Speed (m/s)
            speed = dist_meters / dt

            # Convert to km/h
            speed_kmh = speed * 3.6

            cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        prev_positions[car_id] = (cx, cy)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Car Speed Detection", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
