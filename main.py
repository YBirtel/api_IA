import json
from unittest import result
import joblib
import pandas as pd
from IA import predictOffer, setupIA

from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
# Route de l'IA
@app.route('/', methods=('GET', 'POST'))
def predict():
    result = ""
    if request.form.get('test') == "1":

        # Récupération des données du formulaire

        manufacturer = request.form.get("manufacturer")
        model = request.form.get("model")
        transmission = request.form.get("transmission")
        color = request.form.get("color")
        odometer = request.form.get("odometer")
        year = request.form.get("year")
        fuel = request.form.get("fuel")
        engine = request.form.get("engine")

        # Création du dataFrame avec les données récoltées

        offer_df = pd.DataFrame({'manufacturer_name': [manufacturer], 'model_name': [model], 'transmission': [transmission], 'color': [color], 'odometer_value': [odometer], 'year_produced': [year], 'engine_fuel': [fuel], 'engine_type': [engine]})


        str_price = request.form.get('price')
        price = float(str_price)

        # Prédiction du prix de l'offre récupérée

        prediction = mod_fit.predict(offer_df)
        prediction_price = prediction[0]

        # Calcul du score de fiabilité

        trust_score = predictOffer(price, prediction_price)
        trust_percent = round(trust_score*100,2)

        result = "Votre offre est fiable à " + str(trust_percent) + "%"

    # Affichage du résulat en dessous du formulaire

    return render_template('index.html', data={'test': 1, 'result': result})

if __name__ == '__main__':
    setupIA()
    mod_fit =  joblib.load('model_fit.pkl')
    app.run(port=8080)