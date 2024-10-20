# Loading the data
StudentData = pd.read_csv('Path to the file')

# Fill missing values
StudentData.fillna(0, inplace=True)

# OneHotEncode categorical variables
categorical_columns = ['schoolsup', 'higher']
numerical_columns = ['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']

# Calculate average grade
columns_to_average = ['G1', 'G2', 'G3']
StudentData['Average_Grade'] = StudentData[columns_to_average].mean(axis=1)

# Create categorical grade feature
StudentData['Average_Grade_Cat_1'] = pd.cut(StudentData['Average_Grade'], bins=[0, 10, 20], labels=[0, 1], include_lowest=True)

# Define feature set X and target variable y
X = StudentData[numerical_columns + categorical_columns]
y = StudentData['Average_Grade_Cat_1']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pre-processing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='if_binary'), categorical_columns)
    ])
