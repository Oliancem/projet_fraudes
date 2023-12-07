import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Charger le modèle à partir du fichier
loaded_model = joblib.load('model_random_forest.pkl')

# Lire les nouvelles données
new_data = pd.read_csv('CSV/principal.csv')

# Supprimer les colonnes non nécessaires
new_data = new_data.drop(columns=['nameOrig', 'nameDest', 'isFraud'])

# Encodez la variable 'type'
label_encoder = LabelEncoder()
new_data['type'] = label_encoder.fit_transform(new_data['type'])

# Imputez les valeurs manquantes
imputer = SimpleImputer(strategy='mean')  # Assurez-vous que cette stratégie correspond à celle utilisée dans le script d'entraînement
new_data = pd.DataFrame(imputer.fit_transform(new_data), columns=new_data.columns)

# Prédictions sur les nouvelles données
predictions = loaded_model.predict(new_data)

# Ajouter les prédictions au DataFrame
new_data['predictions'] = predictions

# Enregistrer les résultats dans un fichier CSV
new_data.to_csv('predictions_fraude.csv', index=False)