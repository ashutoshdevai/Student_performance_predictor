
# ============================================
# STUDENT PERFORMANCE PREDICTOR
# Python + Machine Learning + Streamlit
# ============================================

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# 2. TITLE
# --------------------------------------------

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's expected final score using "
    "academic and lifestyle factors."
)


# 3. CREATE DATASET
# --------------------------------------------

data = pd.DataFrame({
    "study_hours": [
        2, 3, 4, 5, 6, 1.5, 7, 3.5, 4.5, 2.5,
        5.5, 6.5, 1, 3, 4, 5, 2, 7.5, 3.5, 4.5
    ],

    "attendance": [
        65, 72, 78, 85, 90, 60, 95, 75, 82, 70,
        88, 92, 55, 68, 80, 86, 62, 98, 77, 84
    ],

    "assignments_completed": [
        4, 5, 7, 8, 9, 3, 10, 6, 8, 5,
        9, 10, 2, 5, 7, 8, 3, 10, 6, 7
    ],

    "previous_score": [
        55, 60, 68, 74, 82, 50, 90, 65, 72, 58,
        78, 85, 45, 62, 70, 76, 52, 94, 67, 73
    ],

    "sleep_hours": [
        5.5, 6, 6.5, 7, 7.5, 5, 8, 7, 6.5, 6,
        7.5, 8, 5.5, 6.5, 7, 7.5, 5, 8.5, 6, 7
    ],

    "final_score": [
        58, 64, 72, 80, 88, 52, 96, 70, 77, 62,
        85, 92, 48, 65, 75, 82, 54, 99, 71, 79
    ]
})


# 4. DISPLAY DATASET
# --------------------------------------------

st.subheader("📊 Dataset")

st.dataframe(
    data,
    use_container_width=True
)

st.caption(
    "Note: This is a small sample dataset created for demonstration."
)


# 5. SELECT FEATURES AND TARGET
# --------------------------------------------

features = [
    "study_hours",
    "attendance",
    "assignments_completed",
    "previous_score",
    "sleep_hours"
]

X = data[features]
y = data["final_score"]


# 6. SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 7. CREATE AND TRAIN MODEL
# --------------------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# 8. TEST MODEL
# --------------------------------------------

predictions = model.predict(X_test)


# 9. MODEL EVALUATION
# --------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


# 10. DISPLAY MODEL PERFORMANCE
# --------------------------------------------

st.subheader("🤖 Model Performance")

col1, col2 = st.columns(2)

col1.metric(
    "Mean Absolute Error (MAE)",
    f"{mae:.2f}"
)

col2.metric(
    "R² Score",
    f"{r2:.2f}"
)

st.caption(
    "MAE measures average prediction error in score points. "
    "R² indicates how well the model fits the test data; "
    "a negative value is possible."
)


# 11. USER INPUT SECTION
# --------------------------------------------

st.sidebar.header("📝 Enter Student Details")

study_hours = st.sidebar.slider(
    "Daily Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=40,
    max_value=100,
    value=80
)

assignments = st.sidebar.slider(
    "Assignments Completed",
    min_value=0,
    max_value=10,
    value=7
)

previous_score = st.sidebar.slider(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=70
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    min_value=3.0,
    max_value=10.0,
    value=7.0,
    step=0.5
)


# 12. CREATE INPUT DATAFRAME
# --------------------------------------------

input_data = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "assignments_completed": [assignments],
    "previous_score": [previous_score],
    "sleep_hours": [sleep_hours]
})

# Ensure feature order matches the training data.
input_data = input_data[features]


# 13. PREDICT FINAL SCORE
# --------------------------------------------

raw_prediction = model.predict(input_data)[0]

# Keep the predicted score within the valid range.
prediction = max(
    0.0,
    min(100.0, float(raw_prediction))
)


# 14. DISPLAY PREDICTION
# --------------------------------------------

st.subheader("🎯 Prediction")

st.metric(
    "Expected Final Score",
    f"{prediction:.2f}/100"
)


# 15. PERFORMANCE MESSAGE
# --------------------------------------------

if prediction >= 80:
    st.success(
        "Excellent predicted performance! 🎉"
    )

elif prediction >= 60:
    st.info(
        "Good predicted performance. "
        "There is still room for improvement."
    )

else:
    st.warning(
        "The prediction suggests that additional "
        "academic support or study may be helpful."
    )


# 16. SHOW STUDENT INFORMATION
# --------------------------------------------

st.subheader("📋 Student Information")

summary = pd.DataFrame({
    "Factor": [
        "Study Hours",
        "Attendance",
        "Assignments Completed",
        "Previous Score",
        "Sleep Hours"
    ],

    "Value": [
        f"{study_hours} hours/day",
        f"{attendance}%",
        str(assignments),
        f"{previous_score}/100",
        f"{sleep_hours} hours"
    ]
})

st.table(summary)


# 17. DISPLAY LINEAR REGRESSION COEFFICIENTS
# --------------------------------------------

st.subheader("📈 Model Coefficients")

coefficient_data = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

coefficient_data["Coefficient"] = (
    coefficient_data["Coefficient"].round(4)
)

coefficient_data = coefficient_data.sort_values(
    by="Coefficient",
    ascending=False
)

st.dataframe(
    coefficient_data,
    use_container_width=True
)

st.caption(
    "Coefficients show the estimated change in predicted score "
    "for a one-unit increase in a feature, holding other features "
    "constant. They are not directly comparable as feature "
    "importance because the features use different scales."
)

# ============================================
# END OF PROJECT
# ============================================

