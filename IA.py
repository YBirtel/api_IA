import os
import joblib
import pandas as pd
from sklearn import ensemble
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor


def setupIA():

    #import du fichier csv

    data = pd.read_csv('carsTDIA.csv', sep=';')


    # Conversion de données en float

    data['year_produced'] = data['year_produced'].map(lambda x : float(x))

    # Création du modèle

    data_x = data.drop(['price_usd'], axis=1).copy()
    data_y = data['price_usd'].values.copy()

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)


    # Conversion des colonnes à valeur en string

    cat_ft = ['manufacturer_name', 'model_name', 'transmission', 'color', 'engine_fuel']

    temp = ColumnTransformer([('cat', OneHotEncoder(), cat_ft)]).fit(data_x)
    cats = temp.named_transformers_['cat'].categories_

    cat_tr = OneHotEncoder(categories=cats, sparse=False)

    # Colonnes à valeur numériques

    num_ft = ['odometer_value', 'year_produced']
    num_tr = StandardScaler()

    # Standardisation des données

    data_tr = ColumnTransformer([('num', num_tr, num_ft), ('cat', cat_tr, cat_ft)])

    mod = Pipeline([
        ('colonne', data_tr),
        ('model', ExtraTreesRegressor())
    ])

    # Export du modèle fit dans un fichier .pkl

    if not os.path.isfile('model_fit.pkl'):
        mod_fit = mod.fit(x_train, y_train)
        joblib.dump(mod_fit, 'model_fit.pkl')


def predictOffer(price: float, prediction: float):

    # Calcul du score de fiabilité
    trust_score = float(1.0 - abs(price-prediction) / max(price, prediction))
    return trust_score



