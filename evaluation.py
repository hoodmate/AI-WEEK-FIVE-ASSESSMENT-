from sklearn.metrics import accuracy_score, recall_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), recall_score(y_test, preds)
