# New student instances
new_students = pd.DataFrame('path to new students data')

# Predict outcomes for new students
new_students_pred = pipeline.predict(new_students)

# Map predictions to human-readable format
outcome_mapping = {0: 'Fail', 1: 'Succeed'}
new_students['Prediction'] = pd.Series(new_students_pred).map(outcome_mapping)

# Display the predictions
print(new_students[['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime', 'schoolsup', 'higher', 'Prediction']])

# Inform the user about their likely outcome
for index, row in new_students.iterrows():
    if row['Prediction'] == 'Succeed':
        print(f"Student {index + 1}: You are likely to succeed as a student.")
    else:
        print(f"Student {index + 1}: You are likely to fail as a student.")