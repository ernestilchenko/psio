import cv2
import easyocr
import numpy as np


def odczytaj_tablice_rejestracyjna(sciezka_obrazu):
    # Inicjalizacja czytnika
    czytnik = easyocr.Reader(['en'])

    # Wczytanie obrazu
    obraz = cv2.imread(sciezka_obrazu)
    if obraz is None:
        raise ValueError("Nie udało się załadować obrazu")

    # Konwersja do skali szarości
    szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)

    # Progowanie adaptacyjne
    binarna = cv2.adaptiveThreshold(
        szary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Znalezienie konturów
    kontury, _ = cv2.findContours(binarna, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kopia obrazu do rysowania wyników
    obraz_wynikowy = obraz.copy()

    # Lista na wykryty tekst
    wykryty_tekst = []

    # Rozpoznawanie tekstu
    wyniki = czytnik.readtext(obraz)

    for (ramka, tekst, pewnosc) in wyniki:
        if pewnosc > 0.2 and len(tekst) > 4:
            print(f"Znaleziony tekst: {tekst} (pewność: {pewnosc:.2f})")
            wykryty_tekst.append(tekst)

            punkty = np.array(ramka, np.int32)
            cv2.polylines(obraz_wynikowy, [punkty], True, (0, 255, 0), 2)

            cv2.putText(obraz_wynikowy, tekst, (int(ramka[0][0]), int(ramka[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Zapisanie wyników
    cv2.imwrite('wynik_z_ramkami.jpg', obraz_wynikowy)
    cv2.imwrite('przetworzone.jpg', binarna)

    return wykryty_tekst


if __name__ == "__main__":
    sciezka_obrazu = "test/IMAGE 2025-01-18 22:35:36.jpg"
    try:
        wykryty_tekst = odczytaj_tablice_rejestracyjna(sciezka_obrazu)
        print("\nWszystkie znalezione tablice:", wykryty_tekst)
        print("Wyniki zapisano w pliku 'wynik_z_ramkami.jpg'")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")