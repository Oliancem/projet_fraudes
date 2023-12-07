import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
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

# Entraînement du modèle de régression Lasso
model_lasso_regression = Lasso()
model_lasso_regression.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_lasso_regression = model_lasso_regression.predict(X_test)

# Évaluation du modèle
mse_lasso_regression = mean_squared_error(y_test, y_pred_lasso_regression)
print("Mean Squared Error (Lasso Regression):", mse_lasso_regression)
