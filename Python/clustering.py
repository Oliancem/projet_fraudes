import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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

# Entraînement du modèle de clustering hiérarchique
model_agg_clustering = AgglomerativeClustering(n_clusters=2)
model_agg_clustering.fit(X_train)

# Sauvegarde du modèle
joblib.dump(model_agg_clustering, 'model_agg_clustering.pkl')


# Les modèles de clustering, comme l'agglomération hiérarchique, ne sont pas adaptés à la détection de fraudes où vous avez des données étiquetées (fraude ou non-fraude).