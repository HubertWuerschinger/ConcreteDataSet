import streamlit as st
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import numpy as np

# Laden des trainierten Modells
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

def predict_with_model(model, input_values):
    """
    Führt eine Vorhersage mit dem gegebenen Modell und den Eingabewerten durch.

    :param model: Das trainierte Modell.
    :param input_values: Die Eingabewerte als Liste oder Array.
    :return: Die Vorhersage des Modells.
    """
    X = np.array([input_values])
    prediction = model.predict(X)
    return prediction

def main():
    st.title("Meine Streamlit App")
    st.header("Willkommen auf der Hauptseite Test")
    
    # Display some test text
    st.write("Das ist ein Testtext auf der Hauptseite der Streamlit-App.")

    # Abschnitt für SelectSlider-Elemente
    st.header("Materialauswahl für Baumaterialien")

    # Variablen und ihre Bereichsgrenzen
    variables = {
        "cement": (100, 500),
        "slag": (0, 200),
        "flyash": (0, 200),
        "water": (100, 300),
        "superplasticizer": (0, 30),
        "coarseaggregate": (800, 1200),
        "fineaggregate": (600, 1000),
        "age": (1, 365)
    }

    values = []
    for var, (min_val, max_val) in variables.items():
        value = st.select_slider(f"{var.capitalize()} (Einheit)", range(min_val, max_val + 1))
        values.append(value)

    # Vorhersage-Button und Ausgabefeld
    if st.button("Vorhersage machen"):
        # Vorhersage mit dem Modell machen
        prediction = predict_with_model(model, values)
        # Anzeige der Vorhersage in einem Ausgabefeld
        st.write("Vorhersageergebnis:")
        st.text_area("Ergebnis", f"{prediction}", height=100)

if __name__ == "__main__":
    main()
