import cv2
from skimage.filters import try_all_threshold, threshold_li
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('test/parking_video.mp4')
frame_no = 10
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
res, obraz = cap.read()

cv2.imshow('ramka', obraz)
cv2.waitKey(0)

frame_no = 113
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
res, obraz = cap.read()

cv2.imshow('ramka', obraz)
cv2.waitKey(0)

# frame_no = 80
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
res, obraz = cap.read()

cv2.imshow('ramka', obraz)
cv2.waitKey(0)

# import cv2
# import numpy as np

# Wczytaj obraz
image = obraz

# Konwersja na skalę szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# Progowanie obrazu (binaryzacja)
thresh_value = threshold_li(gray)
print(thresh_value)
_, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)

# Zdefiniuj miejsca parkingowe (x, y, szerokość, wysokość)
parking_spots = [
    (0, 90, 310, 130),  # Przykładowe współrzędne miejsca 1
    (0, 220, 310, 145),  # Przykładowe współrzędne miejsca 1
    (0, 365, 310, 152),  # Przykładowe współrzędne miejsca 2
    (0, 517, 310, 140),  # itd.
    (0, 657, 310, 143),  # itd.
    (0, 800, 310, 150),  # itd.
]

# Sprawdź zajętość miejsc parkingowych
for idx, (x, y, w, h) in enumerate(parking_spots):
    roi = thresh[y:y + h, x:x + w]  # Region of Interest
    non_zero = cv2.countNonZero(roi)  # Liczba białych pikseli w obszarze
    total_pixels = w * h

    print(idx, non_zero, total_pixels)

    # Jeśli liczba białych pikseli przekracza próg, oznacz jako zajęte
    if non_zero > total_pixels * 0.3:  # 20% obszaru zajęte
        status = "Zajete"
        color = (0, 0, 255)  # Czerwony
    elif non_zero > total_pixels * 0.1:  # 20% obszaru zajęte
        status = "Zablokowane"
        color = (0, 160, 255)  # Pomarańczowy
    else:
        status = "Wolne"
        color = (0, 255, 0)  # Zielony

    # Narysuj prostokąt i status miejsca na obrazie
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Wyświetl wynik
cv2.imshow("Parking Status", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

