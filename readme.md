Titanic Data Analysis – Kaggle (train.csv)

📌 Opis projektu

Ten projekt to analiza danych pasażerów Titanica z wykorzystaniem pliku train.csv pobranego z Kaggle. Celem jest eksploracja danych, czyszczenie, tworzenie nowych cech oraz wizualizacja zależności wpływających na przeżywalność.

🧰 Wykorzystane technologie

Python 3

Pandas

Seaborn

Matplotlib

NumPy

📂 Dane wejściowe

Plik train.csv z konkursu Titanic na Kaggle.

📊 Zakres analizy

Wczytanie i wstępna eksploracja danych

Czyszczenie brakujących wartości (Age, Embarked, Cabin)

Tworzenie zmiennych pomocniczych:

Title – tytuł pasażera (np. Mr, Miss, etc.)

FamilySize – rozmiar rodziny na pokładzie

IsAlone – czy pasażer podróżował sam

IsChild – czy pasażer był dzieckiem (< 12 lat)

Analiza przeżywalności względem różnych zmiennych

Wizualizacje z wykorzystaniem Seaborn

Eksport oczyszczonych danych do cleaned_train.csv

🏁 Uruchomienie projektu

Upewnij się, że masz zainstalowane biblioteki:

pip install pandas seaborn matplotlib

Umieść plik train.csv w katalogu projektu

Uruchom notebook/skrypt Python:

python titanic_analysis.py

Wyniki znajdziesz w pliku cleaned_train.csv oraz na wykresach.

✅ Efekty

Zidentyfikowano wpływ takich cech jak płeć, klasa, samotność czy wiek na przeżywalność.

Stworzono nowy zbiór danych z dodatkowymi zmiennymi do dalszej analizy lub modelowania.

🧠 Inspiracje

Projekt inspirowany klasycznym problemem Kaggle: "Titanic – Machine Learning from Disaster".

📎 Autor: [Twoje Imię]📅 Rok: 2025

