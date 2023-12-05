import streamlit as st
import pickle
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))


def main():
    # Set up the main page
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

    # Erstellung der SelectSlider für jede Variable
    for var, (min_val, max_val) in variables.items():
        selected_value = st.select_slider(f"{var.capitalize()} (Einheit)", range(min_val, max_val + 1))
        st.write(f"Sie haben {selected_value} als Wert für {var} gewählt.")

if __name__ == "__main__":
    main()
