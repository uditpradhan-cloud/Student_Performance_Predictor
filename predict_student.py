import pandas as pd
import joblib


# Load trained AI model
model = joblib.load("student_performance_model_final.pkl")

print("========================================")
print("     AI STUDENT PERFORMANCE PREDICTOR")
print("========================================")
print()

student = {}

# Student information
student["school"] = input("School (GP/MS): ")
student["sex"] = input("Gender (M/F): ")
student["age"] = int(input("Age: "))
student["address"] = input("Address (U/R): ")
student["famsize"] = input("Family size (GT3/LE3): ")
student["Pstatus"] = input("Parent status (A/T): ")

# Parent information
student["Medu"] = int(input("Mother's education (0-4): "))
student["Fedu"] = int(input("Father's education (0-4): "))

student["Mjob"] = input("Mother's job: ")
student["Fjob"] = input("Father's job: ")

# School information
student["reason"] = input("Reason for choosing school: ")
student["guardian"] = input("Guardian: ")

# Academic / behavioral information
student["traveltime"] = int(input("Travel time (1-4): "))
student["studytime"] = int(input("Study time (1-4): "))
student["failures"] = int(input("Past failures (0-4): "))

# Support and activities
student["schoolsup"] = input("Extra school support (yes/no): ")
student["famsup"] = input("Family support (yes/no): ")
student["paid"] = input("Extra paid classes (yes/no): ")
student["activities"] = input("Extra-curricular activities (yes/no): ")
student["nursery"] = input("Attended nursery school (yes/no): ")
student["higher"] = input("Wants higher education (yes/no): ")
student["internet"] = input("Internet access (yes/no): ")
student["romantic"] = input("In a romantic relationship (yes/no): ")

# Lifestyle information
student["famrel"] = int(input("Family relationship quality (1-5): "))
student["freetime"] = int(input("Free time (1-5): "))
student["goout"] = int(input("Going out frequency (1-5): "))
student["Dalc"] = int(input("Workday alcohol consumption (1-5): "))
student["Walc"] = int(input("Weekend alcohol consumption (1-5): "))
student["health"] = int(input("Current health (1-5): "))
student["absences"] = int(input("Number of absences: "))


# Convert input into DataFrame
student_df = pd.DataFrame([student])


# Make prediction
prediction = model.predict(student_df)[0]


# Display prediction result
print()
print("========================================")
print("          AI PREDICTION RESULT")
print("========================================")
print(f"Predicted Final Grade (G3): {prediction:.2f} / 20")

# Performance level
if prediction >= 16:
    level = "Excellent"
elif prediction >= 12:
    level = "Good"
elif prediction >= 10:
    level = "Average"
else:
    level = "Needs Improvement"

print(f"Performance Level: {level}")


# ========================================
# MODEL INSIGHTS
# ========================================

print()
print("========================================")
print("           MODEL INSIGHTS")
print("========================================")

print("Model: Random Forest Regression")
print("Evaluation: 5-Fold Cross-Validation")
print()

print("Model Performance:")
print("• Mean Absolute Error (MAE): 2.99 marks")
print("• Root Mean Squared Error (RMSE): 4.00 marks")
print("• R² Score: 0.216")
print("• Predictions within ±2 marks: 46.33%")

print()
print("What does this mean?")
print("• Average prediction error is about 3 marks.")
print("• About 46% of predictions are within ±2 marks.")
print("• The model has limited-to-moderate predictive power.")
print("• Prediction is an estimate, not a guaranteed final grade.")

print()
print("Top Factors Considered by the Model:")
print("1. Absences")
print("2. Past failures")
print("3. Health")
print("4. Going out frequency")
print("5. Age")
print("6. Study time")

print()
print("Note:")
print("Feature importance shows which factors the model")
print("relied on most across the dataset.")
print("It does NOT prove that a factor directly causes")
print("a student's final grade.")

print("========================================")