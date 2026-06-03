import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from think.split import split


def evaluate():
    _, X_test, _, y_test = split()

    model = joblib.load('models/best_model.pkl')
    le = joblib.load('models/label_encoder.pkl')

    y_pred = model.predict(X_test)

    labels = le.classes_
    print("Classes:", labels)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to models/confusion_matrix.png")

    print("\n--- Banking FP/FN Analysis ---")
    print("""
False Negative (FN) = Predicting 'Approved' for someone who should be 'Denied'
  → Bank lends money to a bad applicant → financial loss, default risk
  → This is the WORST case in banking

False Positive (FP) = Predicting 'Denied' for someone who should be 'Approved'
  → Bank loses a good customer → missed revenue
  → Bad, but recoverable

To minimize FN, the model was tuned using:
  - scoring='recall_macro' in GridSearchCV
  - class_weight='balanced' to penalize misclassifying minority classes
  - SMOTE to oversample underrepresented classes before training
""")

    denied_idx = list(le.classes_).index('Denied')
    fn = cm[:, denied_idx].sum() - cm[denied_idx, denied_idx]
    fp = cm[denied_idx, :].sum() - cm[denied_idx, denied_idx]
    print(f"False Negatives (should be Denied, predicted otherwise): {fn}")
    print(f"False Positives (predicted Denied, actually not):        {fp}")


if __name__ == '__main__':
    evaluate()