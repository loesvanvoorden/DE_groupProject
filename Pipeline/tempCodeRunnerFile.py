# Define the base models
svm_linear_clf = SVC(C=0.1, kernel='linear', gamma='scale', class_weight='balanced', degree=2, probability=True)
svm_rbf_clf = SVC(C=0.1, kernel='rbf', gamma='scale', class_weight='balanced', degree=2, probability=True)

# Meta-classifier
logreg = LogisticRegression(C=10, solver='newton-cg')

# Stacking ensemble classifier configuration
stack_clf = StackingClassifier(
    estimators=[
        ('svm_linear_clf', svm_linear_clf),
        ('svm_rbf', svm_rbf_clf)
    ],
    final_estimator=logreg,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stack_clf)
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Predict on the training set
y_train_pred = pipeline.predict(X_train)

# Predict on the test set
y_test_pred = pipeline.predict(X_test)