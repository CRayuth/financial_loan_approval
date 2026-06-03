from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sense.preprocess import preprocess


def split(filepath='data/loan_data.csv', test_size=0.2, random_state=42):
    X, y = preprocess(filepath)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size:  {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    split()