import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('power.csv')

st.sidebar.title("Power System")

user_menu = st.sidebar.radio(
    'Select an Option',
    ('OverView','Classification of Faults','Fault Detection')
)
#"https://electrical-engineering-portal.com/wp-content/uploads/2017/10/electric-power-system.png"
if user_menu == 'OverView':
    st.sidebar.header("OverView")
    st.header("Power System Faults Overview")
    img = st.image('powersystem.webp')  
    
    st.write("Power System Fault can be caused by a variety factors, including weather conditions, equipment failure, and human error")
    st.write("Line to line Overview : ") 
    print("\n")
    st.write("Line-to-line faults in power lines occur due to several reasons, including:  1.Physical damage from falling trees, strong winds, or wildlife contact. 2. Insulation breakdown caused by aging, wear, or moisture accumulation.  3 .Lightning strikes, which can cause voltage surges and flashovers between lines.4. Mechanical failures in connectors or components, leading to arcing between phases.5. Overloading or short circuits due to faulty equipment or improper operation.")
    print("\n\n")
    st.write("Line to ground Overview : ") 
    st.write("A line-to-ground fault in a power line occurs due to the following reasons: 1. Weather Conditions: Lightning strikes, strong winds, or heavy rain can damage the lines and cause direct contact with the ground. 2. Physical Damage: Trees falling on power lines or vehicles hitting utility poles can create a fault by bringing lines into contact with the earth. 3. Corrosion: Rust or corrosion in the equipment can weaken the system and lead to ground faults. 4. Insulation Breakdown: Over time, insulation material degrades, leading to exposed conductors, which can come into contact with the ground. 5. Wildlife: Animals such as birds or squirrels making contact with the line can create a ground fault.")

if user_menu == 'Classification of Faults':
    st.sidebar.header("Classification of Faults")
    st.write("Power System Fault can be caused by a variety factors, including weather conditions, equipment failure, and human error")
    st.dataframe(df)
    st.write("Line to Line Fault.")
    img2 = st.image('line-to-line-fault.jpg')
    st.write("Condition: If two phases show abnormally high or low current while the third phase is normal, it could indicate a fault between the two lines.")
    st.write("Ground Detection : ")
    st.write("Condition: If the current in one phase is significantly higher than normal, while the other phases are normal or low, it indicates a fault between the phase and ground.")
    st.write("Three-Phase Fault (LLL Fault): ")
    st.write("Condition: If all three phases show abnormal readings, it could indicate a three-phase fault.")
    
if user_menu == 'Fault Detection':
    st.sidebar.header("Fault Detection")
    df = pd.read_csv('classData.csv')

    # Features (current and voltage values)
    X = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]

    # Target (fault labels G, C, B, A)
    y = df[['G', 'C', 'B', 'A']]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Streamlit App
    st.title("Fault Detection in Power Systems")

    # Input fields for current and voltage values
    Ia = st.number_input("Enter Ia (Current in Phase A):", value=100.0)
    Ib = st.number_input("Enter Ib (Current in Phase B):", value=90.0)
    Ic = st.number_input("Enter Ic (Current in Phase C):", value=85.0)
    Va = st.number_input("Enter Va (Voltage in Phase A):", value=220.0)
    Vb = st.number_input("Enter Vb (Voltage in Phase B):", value=230.0)
    Vc = st.number_input("Enter Vc (Voltage in Phase C):", value=240.0)

    # Fault conditions dictionary
    fault_conditions = {
        (0, 0, 0, 0): "No Fault",
        (1, 0, 0, 0): "Ground Fault",
        (0, 0, 0, 1): "Fault in Line A",
        (0, 0, 1, 0): "Fault in Line B",
        (0, 1, 0, 0): "Fault in Line C",
        (1, 0, 0, 1): "LG Fault (Between Phase A and Ground)",
        (1, 0, 1, 0): "LG Fault (Between Phase B and Ground)",
        (1, 1, 0, 0): "LG Fault (Between Phase C and Ground)",
        (0, 0, 1, 1): "LL Fault (Between Phase B and A)",
        (0, 1, 1, 0): "LL Fault (Between Phase C and B)",
        (0, 1, 0, 1): "LL Fault (Between Phase C and A)",
        (1, 0, 1, 1): "LLG Fault (Between Phases A, B, and Ground)",
        (1, 1, 0, 1): "LLG Fault (Between Phases A, C, and Ground)",
        (1, 1, 1, 0): "LLG Fault (Between Phases B, C, and Ground)",
        (0, 1, 1, 1): "LLL Fault (Between all three phases)",
        (1, 1, 1, 1): "LLLG Fault (Three-phase symmetrical fault)"
    }

    # Prediction on the entered values
    if st.button("Predict Fault"):
        data = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])
        predicted_fault = model.predict(data)

        # Convert predicted fault (numpy array) to tuple
        predicted_fault_tuple = tuple(predicted_fault[0])

        # Get the fault condition
        fault_name = fault_conditions.get(predicted_fault_tuple, "Unknown Fault")

        # Show the predicted fault
        st.write(f"Predicted Fault: {fault_name}")

        # Model accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    feature_importances = model.feature_importances_

    # ग्राफ़ का निर्माण करें
    fig, ax = plt.subplots()
    ax.barh(X.columns, feature_importances)
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance Graph')

    # Streamlit में ग्राफ़ प्रदर्शित करें
    st.pyplot(fig)

    fig2,ax2 = plt.subplots()
    currents = [Ia, Ib, Ic]
    voltage = [Va, Vb, Vc]
    phases = ["A", "B", "C"]

    ax2.plot(phases, currents, label="current(A)", marker = "o")
    ax2.plot(phases, voltage, label="voltage(V)", marker = "x")

    ax2.set_xlabel('phases')
    ax2.set_ylabel('values')
    ax2.set_title('Current vs Voltage by Phase')
    ax2.legend()

    # Display the Current vs Voltage graph in Streamlit
    st.pyplot(fig2)