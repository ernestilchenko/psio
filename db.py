import pymongo
from datetime import datetime
import cv2
import easyocr
import time
from enum import Enum
import RPi.GPIO as GPIO


# Stany bramki
class StanBramki(Enum):
    ZAMKNIETA = 0
    OTWIERANIE = 1
    OTWARTA = 2
    ZAMYKANIE = 3


class KontrolerBramki:
    def __init__(self):
        self.stan = StanBramki.ZAMKNIETA
        self.czas_otwarcia = None
        # Sensory
        self.SENSOR_PRZED_BRAMKA = 17  # GPIO pin
        self.SENSOR_POD_BRAMKA = 18  # GPIO pin
        self.SENSOR_ZA_BRAMKA = 19  # GPIO pin

        # Konfiguracja GPIO (symulacja)
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(self.SENSOR_PRZED_BRAMKA, GPIO.IN)
        # GPIO.setup(self.SENSOR_POD_BRAMKA, GPIO.IN)
        # GPIO.setup(self.SENSOR_ZA_BRAMKA, GPIO.IN)

        self.MAX_CZAS_OTWARCIA = 30  # seconds
        self.CZAS_PELNEGO_OTWARCIA = 3  # seconds

    def sprawdz_pojazd_przed_bramka(self):
        # return GPIO.input(self.SENSOR_PRZED_BRAMKA)
        return True  # Symulacja

    def sprawdz_pojazd_pod_bramka(self):
        # return GPIO.input(self.SENSOR_POD_BRAMKA)
        return False  # Symulacja

    def sprawdz_pojazd_za_bramka(self):
        # return GPIO.input(self.SENSOR_ZA_BRAMKA)
        return False  # Symulacja

    def otworz_bramke(self):
        if self.stan == StanBramki.ZAMKNIETA:
            print("Otwieranie bramki...")
            self.stan = StanBramki.OTWIERANIE
            time.sleep(self.CZAS_PELNEGO_OTWARCIA)  # Symulacja czasu otwierania
            self.stan = StanBramki.OTWARTA
            self.czas_otwarcia = datetime.now()
            return True
        return False

    def zamknij_bramke(self):
        if self.stan == StanBramki.OTWARTA:
            if not self.sprawdz_pojazd_pod_bramka():
                print("Zamykanie bramki...")
                self.stan = StanBramki.ZAMYKANIE
                time.sleep(self.CZAS_PELNEGO_OTWARCIA)  # Symulacja czasu zamykania
                self.stan = StanBramki.ZAMKNIETA
                self.czas_otwarcia = None
                return True
        return False

    def sprawdz_timeout(self):
        if self.stan == StanBramki.OTWARTA and self.czas_otwarcia:
            czas_otwarcia = (datetime.now() - self.czas_otwarcia).total_seconds()
            return czas_otwarcia > self.MAX_CZAS_OTWARCIA
        return False


class SystemParkingowy:
    def __init__(self):
        # Połączenie z MongoDB
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["parking_system"]
        self.authorized_plates = self.db["authorized_plates"]
        self.parking_status = self.db["parking_status"]
        self.kontroler_bramki = KontrolerBramki()

        # Inicjalizacja statusu parkingu
        if not self.parking_status.find_one({"_id": "main"}):
            self.parking_status.insert_one({
                "_id": "main",
                "zajete_miejsca": 0,
                "max_miejsc": 6
            })

    def obsługa_wjazdu(self, obraz_path):
        """Obsługa procesu wjazdu z zabezpieczeniami"""
        # Wykrywanie tablicy
        numer_rejestracyjny = self.rozpoznaj_tablice(obraz_path)
        if not numer_rejestracyjny:
            return "Nie wykryto tablicy rejestracyjnej"

        # Sprawdzenie uprawnień
        if not self.sprawdz_uprawnienia(numer_rejestracyjny):
            self.dodaj_log(numer_rejestracyjny, "próba wjazdu", "odmowa - brak uprawnień")
            return "Brak uprawnień"

        # Sprawdzenie dostępności miejsc
        if not self.sprawdz_dostepnosc_miejsc():
            self.dodaj_log(numer_rejestracyjny, "próba wjazdu", "odmowa - brak miejsc")
            return "Brak wolnych miejsc"

        # Proces otwierania bramki i monitorowania przejazdu
        return self.proces_wjazdu(numer_rejestracyjny)

    def proces_wjazdu(self, numer_rejestracyjny):
        """Bezpieczny proces obsługi wjazdu pojazdu"""
        # Otwarcie bramki
        if not self.kontroler_bramki.otworz_bramke():
            return "Błąd otwarcia bramki"

        # Monitoring przejazdu
        start_time = time.time()
        przejazd_zakonczony = False

        while time.time() - start_time < 30:  # Maksymalny czas oczekiwania
            # Sprawdzenie czy pojazd jest pod bramką
            if self.kontroler_bramki.sprawdz_pojazd_pod_bramka():
                # Czekamy aż przejedzie za bramkę
                while self.kontroler_bramki.sprawdz_pojazd_pod_bramka():
                    time.sleep(0.1)
                    if self.kontroler_bramki.sprawdz_timeout():
                        self.dodaj_log(numer_rejestracyjny, "wjazd", "timeout")
                        return "Timeout - przekroczono czas przejazdu"

                # Pojazd przejechał za bramkę
                if self.kontroler_bramki.sprawdz_pojazd_za_bramka():
                    przejazd_zakonczony = True
                    break

            # Sprawdzenie czy pojazd nie wycofał
            if not self.kontroler_bramki.sprawdz_pojazd_przed_bramka():
                self.kontroler_bramki.zamknij_bramke()
                self.dodaj_log(numer_rejestracyjny, "wjazd", "rezygnacja")
                return "Rezygnacja z wjazdu"

            time.sleep(0.1)

        # Zamknięcie bramki i aktualizacja stanu
        if przejazd_zakonczony:
            self.kontroler_bramki.zamknij_bramke()
            self.aktualizuj_status_parkingu("wjazd")
            self.dodaj_log(numer_rejestracyjny, "wjazd", "sukces")
            return "Wjazd zakończony pomyślnie"
        else:
            self.kontroler_bramki.zamknij_bramke()
            self.dodaj_log(numer_rejestracyjny, "wjazd", "timeout")
            return "Timeout - nie zakończono wjazdu"

    def sprawdz_dostepnosc_miejsc(self):
        """Sprawdza czy są wolne miejsca"""
        status = self.parking_status.find_one({"_id": "main"})
        return status["zajete_miejsca"] < status["max_miejsc"]

    def dodaj_log(self, numer_rejestracyjny, akcja, status):
        """Zapisuje log w bazie danych"""
        self.db.parking_logs.insert_one({
            "numer_rejestracyjny": numer_rejestracyjny,
            "akcja": akcja,
            "status": status,
            "data": datetime.now()
        })

    def rozpoznaj_tablice(self, obraz_path):
        """Rozpoznawanie tablicy rejestracyjnej"""
        czytnik = easyocr.Reader(['en'])
        obraz = cv2.imread(obraz_path)
        if obraz is None:
            return None

        wyniki = czytnik.readtext(obraz)
        return wyniki[0][1] if wyniki else None

    def sprawdz_uprawnienia(self, numer_rejestracyjny):
        """Sprawdza uprawnienia pojazdu"""
        return self.authorized_plates.find_one({"numer_rejestracyjny": numer_rejestracyjny}) is not None


# Przykład użycia
if __name__ == "__main__":
    system = SystemParkingowy()

    # Symulacja wjazdu
    print("Symulacja procesu wjazdu:")
    wynik = system.obsługa_wjazdu("sciezka/do/zdjecia.jpg")
    print(f"Rezultat: {wynik}")