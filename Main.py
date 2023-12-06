import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Laden des trainierten Modells
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))


# Funktion zum Laden des Skalierers aus der Pickle-Datei
def load_scaler(filename):
    with open(filename, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Laden des Skalierers für X und Y
scaler_X_filename = 'ScaleFaktorsX.sav'  # Pfad zu Ihrer Pickle-Datei für X
scaler_Y_filename = 'ScaleFaktorsy.sav'  # Pfad zu Ihrer Pickle-Datei für Y
scaler_X = load_scaler(scaler_X_filename)
scaler_Y = load_scaler(scaler_Y_filename)

# Funktion zum Skalieren der Eingabedaten
def scale_input(input_values, scaler):
    X_scaled = scaler.transform(np.array([input_values]))
    return X_scaled  

def predict_with_model(model, input_values, scaler_X):
    input_values_scaled = scale_input(input_values, scaler_X)
    prediction = model.predict(input_values_scaled)
    return prediction

def main():
    st.title("Meine Streamlit App")
    st.header("Willkommen auf der Hauptseite Test")

    # Anzeigen der Skalierungsfaktoren
    st.write("Skalierungsfaktoren und Min-Werte für X:")
    st.write("Skalierungsfaktoren:", scaler_X)
    

    st.write("Skalierungsfaktoren und Min-Werte für Y:")
    st.write("Skalierungsfaktoren:", scaler_Y)
    

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
   # Vorhersage-Button und Ausgabefeld
    if st.button("Vorhersage machen"):
        # Skalierung der Eingabewerte
        input_values_scaled = scale_input(values, scaler_X)

        # Vorhersage mit dem Modell machen
        prediction = model.predict(input_values_scaled)

        # Anzeige der Vorhersage in einem Ausgabefeld
        st.write("Vorhersageergebnis:")
        st.text_area("Ergebnis", f"{prediction}", height=100)


if __name__ == "__main__":
    main()
