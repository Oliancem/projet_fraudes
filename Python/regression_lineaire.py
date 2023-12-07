import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Fonction pour évaluer un modèle et imprimer les résultats
def evaluate_model(model, X_test, y_test):
    if isinstance(model, (LinearRegression)):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)
    else:
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

# Chargement des données
data = pd.read_csv('CSV/principal.csv')

# Nettoyage des données
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

# Réduction de dimension avec PCA
num_components = min(X_train.shape[0], X_train.shape[1]) - 1
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Entraînement du modèle de régression linéaire
model_linear_regression = LinearRegression()
model_linear_regression.fit(X_train_pca, y_train)

# Prédiction sur l'ensemble de test
y_pred_linear_regression = model_linear_regression.predict(X_test_pca)

# Évaluation du modèle
print("Linear Regression Model:")
evaluate_model(model_linear_regression, X_test_pca, y_test)
