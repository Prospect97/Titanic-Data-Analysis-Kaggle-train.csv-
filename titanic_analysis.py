# Titanic Data Analysis Project – Kaggle train.csv version

# 1. Importowanie bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 2. Wczytanie danych
df = pd.read_csv("train.csv")

# 3. Podstawowa eksploracja danych
print(df.head())
print(df.info())
print(df.describe())

# 4. Czyszczenie danych
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')

#Tu coś kernel w jupyterze errorował więc poprawiono na:

# Sprawdzenie ile jest brakujących wartości w każdej kolumnie
print(df.isnull().sum())
# Uzupełniamy brakujące wartości w kolumnie 'Age' medianą wieku i przypisujemy z powrotem do kolumny
df['Age'] = df['Age'].fillna(df['Age'].median())
# Uzupełniamy brakujące wartości w kolumnie 'Embarked' najczęściej występującą wartością (modą) i przypisujemy z powrotem
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# W kolumnie 'Cabin' brakujące wartości zastępujemy tekstem 'Unknown' i przypisujemy z powrotem
df['Cabin'] = df['Cabin'].fillna('Unknown')


# 5. Tworzenie nowych zmiennych
# Tytuł pasażera (Mr, Miss, etc.)
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Rodzina na pokładzie
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Czy pasażer był sam?
df['IsAlone'] = (df['FamilySize'] == 1)

# Czy dziecko?
df['IsChild'] = df['Age'] < 10

# Czy nastolatek?
df['IsTeenager'] = (df['Age'] >= 10) & (df['Age'] <= 19)

# 6. Grupowanie i analiza
print(df.groupby('Pclass')['Survived'].mean())
print(df.groupby('Sex')['Survived'].mean())
print(df.groupby('Title')['Survived'].mean())

# 7. Wizualizacje
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.show()

sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Sex')
plt.show()

sns.countplot(data=df, x='IsAlone', hue='Survived')
plt.title('Survival by Alone Status')
plt.show()

sns.countplot(data= df, x='IsTeenager', hue='Survived')
plt.tile('Survival by Teenagers')

# 8. Eksport przetworzonych danych
df.to_csv("cleaned_train.csv", index=False)





