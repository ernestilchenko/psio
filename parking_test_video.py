import cv2
from skimage.filters import threshold_li

parking_spots = [
    (0, 90, 310, 130),
    (0, 220, 310, 145),
    (0, 365, 310, 152),
    (0, 517, 310, 140),
    (0, 657, 310, 143),
    (0, 800, 310, 150),
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

            # Narysuj prostokąt i status miejsca na obrazie
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Parking Status", frame)

        # Zapisz przetworzony frame do pliku wyjściowego
        out.write(frame)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

cap.release()
out.release()
cv2.destroyAllWindows()