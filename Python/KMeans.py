import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score

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

# Utilisation de RandomOverSampler pour gérer le déséquilibre de classe
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# Entraînement du modèle KMeans
model_kmeans = KMeans(n_clusters=2, random_state=42)
model_kmeans.fit(X_resampled)

# Sauvegarde du modèle
joblib.dump(model_kmeans, 'model_kmeans.pkl')

# KMeans n'est pas un modèle de détection de fraude, il est utilisé pour le clustering.