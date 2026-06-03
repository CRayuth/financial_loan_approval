import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from think.split import split


def train():
    X_train, X_test, y_train, y_test = split()

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ''

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} — tuning now...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'class_weight': ['balanced']
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring='recall_macro',
        cv=3,
        n_jobs=1
    )
    grid.fit(X_train, y_train)

    tuned_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"\nTuned model report:\n")
    print(classification_report(y_test, tuned_model.predict(X_test)))

    joblib.dump(tuned_model, 'models/best_model.pkl')
    print("Model saved to models/best_model.pkl")


if __name__ == '__main__':
    train()