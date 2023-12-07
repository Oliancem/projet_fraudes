import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler

# Chargement des données
data = pd.read_csv('CSV/principal.csv')
data = pd.read_csv()

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

# Assurez-vous que vos données sont dans un format adapté à l'algorithme Apriori
# Supposons que vos données soient dans un format où chaque transaction est une liste d'articles
transactions = data_imputed['transaction'].tolist()

# Utilisation de l'algorithme Apriori
frequent_itemsets = apriori(transactions, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Affichage des règles d'association
print("Association Rules:")
print(rules)

# Malheuresement ici on ne peut pas utiliser Apriori car il est concu pour l'analyse d'articles frequents dans des transactions