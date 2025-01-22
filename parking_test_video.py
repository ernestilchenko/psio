import cv2
from skimage.filters import threshold_li
import easyocr
import numpy as np
import re

def read_registration(image):
    # Inicjalizacja czytnika
    reader = easyocr.Reader(['en'])

    # Rozpoznawanie tekstu
    results = reader.readtext(image)

    pattern = r'[A-Z]{2,3}[A-Z0-9]{2,7}'

    for (ramka, tekst, pewnosc) in results:
        if pewnosc > 0.2 and len(tekst) > 4:

            matches = re.findall(pattern, tekst)
            print(matches)
            if len(matches) > 0:
                return matches[0]

    return None


parking_spots = [
    (0, 90, 310, 130), # id 0
    (0, 220, 310, 145), # id 1
    (0, 365, 310, 152), # id 2
    (0, 517, 310, 140), # id 3
    (0, 657, 310, 143), # id 4
    (0, 800, 310, 150), # id 5
]
cap = cv2.VideoCapture('test/parking_video.mp4')

# Sprawdź właściwości wideo wejściowego
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inicjalizacja VideoWriter do zapisu wyniku
output_file = 'test/processed_parking_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Kodowanie wideo (np. mp4v dla .mp4)
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

parking_spots_status = [["Wolne", None] for _ in range(len(parking_spots))]

while True:
    flag, frame = cap.read()
    if not flag:
        break

    if flag:
        # zamiana na odcienie szarosci
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # progowanie obrazu
        thresh_value = threshold_li(gray)
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)

        for idx, (x, y, w, h) in enumerate(parking_spots):
            roi = thresh[y:y + h, x:x + w]  # Region of Interest
            non_zero = cv2.countNonZero(roi)  # Liczba białych pikseli w obszarze
            total_pixels = w * h # wszystkie piksele

            # Jeśli liczba białych pikseli przekracza próg, oznacz jako zajęte
            if non_zero > total_pixels * 0.3:  # 30% obszaru zajęte
                status = "Zajete"
                color = (0, 0, 255)  # Czerwony
            elif non_zero > total_pixels * 0.1:  # 10% obszaru zajęte
                status = "Zablokowane"
                color = (0, 160, 255)  # Pomarańczowy
            else:
                status = "Wolne"
                color = (0, 255, 0)  # Zielony

            if status != parking_spots_status[idx][0]:
                parking_spots_status[idx][0] = status
                if status == "Wolne":
                    print("Miejsce zostało zwolnine, rejestracja: ", parking_spots_status[idx][1])
                    parking_spots_status[idx][1] = None
                if status == "Zajete":
                    texts = read_registration(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
                    parking_spots_status[idx][1] = texts
                    print("Miejsce zostało zajęte, rejestracja: ", texts)

            # Narysuj prostokąt i status miejsca na obrazie
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Parking Status", frame)

        # Zapisz przetworzony frame do pliku wyjściowego
        out.write(frame)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

print(parking_spots_status)
cap.release()
out.release()
cv2.destroyAllWindows()