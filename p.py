import re
import sqlite3
import time

import cv2
import easyocr
import numpy as np
import torch
from skimage.filters import threshold_li
from ultralytics import YOLO


class ParkingDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('db/parking.db')
        self.cursor = self.conn.cursor()
        self.setup_database()
        self.spot_timers = {}
        self.PARKING_THRESHOLD = 2.0
        self.CLEAR_THRESHOLD = 1.0

    def setup_database(self):
        # Create tables if they don't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS authorized_plates (
            plate_number TEXT PRIMARY KEY
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS parking_status (
            spot_number INTEGER PRIMARY KEY,
            plate_number TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Insert the only authorized plate if it doesn't exist
        self.cursor.execute('''
        INSERT OR IGNORE INTO authorized_plates (plate_number)
        VALUES ('KX 7777 KX')
        ''')
        self.conn.commit()

    def is_plate_authorized(self, plate_number):
        self.cursor.execute('SELECT 1 FROM authorized_plates WHERE plate_number = ?', (plate_number,))
        return self.cursor.fetchone() is not None

    def update_parking_status(self, spot_number, plate_number, current_time):
        if spot_number not in self.spot_timers:
            self.spot_timers[spot_number] = {'plate': plate_number, 'start_time': current_time,
                                             'last_seen': current_time}
        elif self.spot_timers[spot_number]['plate'] == plate_number:
            self.spot_timers[spot_number]['last_seen'] = current_time
            # If the car has been in the spot long enough, update the database
            if current_time - self.spot_timers[spot_number]['start_time'] >= self.PARKING_THRESHOLD:
                self.cursor.execute('''
                INSERT OR REPLACE INTO parking_status (spot_number, plate_number)
                VALUES (?, ?)
                ''', (spot_number, plate_number))
                self.conn.commit()
        else:
            # Different plate detected, reset timer
            self.spot_timers[spot_number] = {'plate': plate_number, 'start_time': current_time,
                                             'last_seen': current_time}

    def check_and_clear_spots(self, current_time):
        spots_to_clear = []
        for spot_number, timer_data in self.spot_timers.items():
            if current_time - timer_data['last_seen'] >= self.CLEAR_THRESHOLD:
                spots_to_clear.append(spot_number)
                self.cursor.execute('DELETE FROM parking_status WHERE spot_number = ?', (spot_number,))

        for spot in spots_to_clear:
            del self.spot_timers[spot]

        if spots_to_clear:
            self.conn.commit()

    def get_parking_status(self):
        self.cursor.execute('SELECT spot_number, plate_number FROM parking_status')
        return self.cursor.fetchall()

    def remove_car(self, plate_number):
        self.cursor.execute('DELETE FROM parking_status WHERE plate_number = ?', (plate_number,))
        self.conn.commit()


class LicensePlateDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def preprocess_plate(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(bilateral, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return thresh

    def find_plate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [location], 0, 255, -1)
            (x, y, w, h) = cv2.boundingRect(location)
            return x, y, w, h

        return None

    def read_plate(self, img):
        processed_img = self.preprocess_plate(img)
        results = self.reader.readtext(processed_img)

        if results:
            text = max(results, key=lambda x: x[2])[1]
            text = re.sub(r'[^A-Za-z0-9]', '', text)
            return text

        return None


def load_models():
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = 'cpu'
        print("Using CPU")

    try:
        plate_model = YOLO('models/labels.pt')
        car_model = YOLO('models/best_car_detection.pt')
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, device

    return car_model, plate_model, device


def process_video(video_path, car_model, plate_model, device, conf_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    db = ParkingDatabase()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    parking_spots = [
        # Top row (left to right)
        (23, 28, 111, 246),
        (141, 24, 108, 249),
        (254, 27, 113, 244),
        (372, 24, 115, 246),
        (496, 21, 108, 244),
        (617, 20, 101, 246),
        # Entry point
        (350, 551, 120, 230),
        # Exit point
        (484, 555, 123, 227),
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/output.mp4', fourcc, fps, (frame_width, frame_height))

    access_message = ""
    access_message_timer = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time
        frame, spots_status = check_parking_spots(frame.copy(), parking_spots)

        # Clear spots that haven't been updated recently
        db.check_and_clear_spots(current_time)

        # Process cars first
        car_results = car_model(frame)[0]
        for car_box in car_results.boxes.data:
            x1, y1, x2, y2, conf, cls = car_box.cpu().numpy()
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Car: {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2)

        # Process plates
        detected_spots = set()
        plate_results = plate_model(frame)[0]

        for r in plate_results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.cpu().numpy()

            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                plate_number = plate_results.names[int(cls)]

                # Check entry point (spot 7)
                if is_in_spot(x1, y1, x2, y2, parking_spots[6]):
                    if db.is_plate_authorized(plate_number):
                        access_message = f"Access granted for {plate_number}"
                        access_message_timer = 50
                    else:
                        access_message = f"Access denied for {plate_number}"
                        access_message_timer = 50

                # Check exit point (spot 8)
                elif is_in_spot(x1, y1, x2, y2, parking_spots[7]):
                    access_message = f"Vehicle {plate_number} has exited"
                    access_message_timer = 50
                    db.remove_car(plate_number)

                # Update regular parking spots
                for idx, spot in enumerate(parking_spots[:6]):
                    if is_in_spot(x1, y1, x2, y2, spot):
                        detected_spots.add(idx + 1)
                        db.update_parking_status(idx + 1, plate_number, current_time)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"Plate: {plate_number}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2)

        # Display access message
        if access_message_timer > 0:
            cv2.putText(frame, access_message,
                        (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if "granted" in access_message else (0, 0, 255),
                        2)
            access_message_timer -= 1

        # Display parking status
        parking_status = db.get_parking_status()
        y_offset = 50
        for spot_num, plate in parking_status:
            if spot_num <= 6:  # Only show regular parking spots
                status_text = f"Spot {spot_num}: {plate}"
                cv2.putText(frame, status_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2)
                y_offset += 30

        out.write(frame)
        cv2.imshow('Parking Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def is_in_spot(x1, y1, x2, y2, spot):
    sx, sy, sw, sh = spot
    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2
    return (center_x >= sx and center_x <= sx + sw and
            center_y >= sy and center_y <= sy + sh)


def check_parking_spots(frame, parking_spots):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_value = threshold_li(gray)
    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)

    spots_status = []

    for idx, (x, y, w, h) in enumerate(parking_spots):
        roi = thresh[y:y + h, x:x + w]
        non_zero = cv2.countNonZero(roi)
        total_pixels = w * h

        if non_zero > total_pixels * 0.3:
            status = "Occupied"
            color = (0, 0, 255)
        elif non_zero > total_pixels * 0.1:
            status = "Blocked"
            color = (0, 160, 255)
        else:
            status = "Free"
            color = (0, 255, 0)

        spots_status.append((status, color))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Spot {idx + 1}: {status}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

    return frame, spots_status


def main():
    VIDEO_PATH = 'test/IMG_6490.MOV'

    car_model, plate_model, device = load_models()
    if car_model is None or plate_model is None:
        print("Failed to load models. Exiting...")
        return

    print(f"Using device: {device}")
    process_video(VIDEO_PATH, car_model, plate_model, device)


if __name__ == '__main__':
    main()
