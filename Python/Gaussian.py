import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler


# Chargement des données
data = pd.read_csv('CSV/principal.csv')

# Suppression des colonnes 'nameOrig' et 'nameDest'
data = data.drop(['nameOrig', 'nameDest'], axis=1)

# Encodage de la variable 'type'
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

# Imputation des valeurs manquantes en utilisant la moyenne
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Arrondir la colonne 'isFraud' au nombre entier le plus proche
data_imputed['isFraud'] = data_imputed['isFraud'].round().astype(int)

# Division des données en ensembles d'entraînement et de test
X = data_imputed.drop('isFraud', axis=1)
y = data_imputed['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de mélange gaussien
model_gaussian_mixture = GaussianMixture(n_components=2, random_state=42)
model_gaussian_mixture.fit(X_train)

# Estimation des densités de probabilité pour les ensembles d'entraînement et de test
train_probabilities = model_gaussian_mixture.score_samples(X_train)
test_probabilities = model_gaussian_mixture.score_samples(X_test)

# Définir un seuil de probabilité pour la classification
threshold = -10 #  les transactions ayant une probabilité en dessous du seuil défini sont considérées comme frauduleuses. Vous devrez ajuster le seuil en fonction de vos besoins et de la distribution de probabilité générée par votre modèle.

# Prédiction sur l'ensemble de test
y_pred_gaussian_mixture = (test_probabilities < threshold).astype(int)

# Évaluation du modèle
print("Gaussian Mixture Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_gaussian_mixture))
print("Classification Report:\n", classification_report(y_test, y_pred_gaussian_mixture))
