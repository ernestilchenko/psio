import re
import cv2
import easyocr
import numpy as np
import torch
from skimage.filters import threshold_li
from ultralytics import YOLO


class LicensePlateDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def preprocess_plate(self, plate_img):
        """Preprocess the license plate image for better OCR"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(bilateral, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return thresh

    def find_plate(self, img):
        """Find potential license plate regions"""
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
        """Read text from license plate"""
        processed_img = self.preprocess_plate(img)
        results = self.reader.readtext(processed_img)

        if results:
            text = max(results, key=lambda x: x[2])[1]
            text = re.sub(r'[^A-Za-z0-9]', '', text)
            return text

        return None


def load_models():
    """Load both YOLO models with proper device handling"""
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load both models
    car_model = YOLO('/Users/ernestilchenko/Downloads/parking/labels.pt')
    plate_model = YOLO('best_car_detection.pt')

    return car_model, plate_model, device


def check_parking_spots(frame, parking_spots):
    """Check if parking spots are occupied"""
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
        cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, spots_status


def process_video(video_path, car_model, plate_model, device, conf_threshold=0.5):
    """Process video using both models"""
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    parking_spots = [
        (0, 90, 310, 130),
        (0, 220, 310, 145),
        (0, 365, 310, 152),
        (0, 517, 310, 140),
        (0, 657, 310, 143),
        (0, 800, 310, 150),
    ]

    plate_detector = LicensePlateDetector()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    spot_plates = {i: None for i in range(len(parking_spots))}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process parking spots
        frame, spots_status = check_parking_spots(frame.copy(), parking_spots)

        # Run both models
        car_results = car_model(frame)[0]
        plate_results = plate_model(frame)[0]

        # Process car detections
        for r in car_results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.cpu().numpy()

            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Plate: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process plate detections
        for r in plate_results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.cpu().numpy()

            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                plate_img = frame[y1:y2, x1:x2]

                # Try to read the plate
                plate_number = plate_detector.read_plate(plate_img)

                if plate_number:
                    # Find corresponding parking spot
                    for idx, (sx, sy, sw, sh) in enumerate(parking_spots):
                        if (x1 + (x2 - x1) // 2 >= sx and
                                x1 + (x2 - x1) // 2 <= sx + sw and
                                y1 + (y2 - y1) // 2 >= sy and
                                y1 + (y2 - y1) // 2 <= sy + sh):
                            spot_plates[idx] = plate_number
                            cv2.putText(frame, f"Plate: {plate_number}",
                                        (sx, sy - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f'Car: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Parking Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    VIDEO_PATH = '/Users/ernestilchenko/Downloads/parking/test/parking_video.mp4'

    car_model, plate_model, device = load_models()
    print(f"Using device: {device}")

    process_video(VIDEO_PATH, car_model, plate_model, device)


if __name__ == '__main__':
    main()