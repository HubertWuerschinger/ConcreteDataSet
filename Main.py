import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Laden des trainierten Modells
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

# Laden des Skalierers
scaler_filename = 'ScaleFaktorsX.sav'  # Pfad zu Ihrer Pickle-Datei
with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

def scale_input(input_values, scaler):
    X_scaled = scaler.transform(np.array([input_values]))
    return X_scaled

def predict_with_model(model, input_values, scaler):
    input_values_scaled = scale_input(input_values, scaler)
    prediction = model.predict(input_values_scaled)
    return prediction

def main():
    st.title("Meine Streamlit App")
    st.header("Willkommen auf der Hauptseite Test")

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
        prediction = predict_with_model(model, values, scaler)
        # Anzeige der Vorhersage in einem Ausgabefeld
        st.write("Vorhersageergebnis:")
        st.text_area("Ergebnis", f"{prediction}", height=100)

if __name__ == "__main__":
    main()
