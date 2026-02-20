import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

def train_model(X_train, y_train):
    """Train a RandomForest model and return it."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    """Compute precision, recall, and F1."""
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    """Run model inferences and return predictions."""
    return model.predict(X)

def save_model(model, path):
    """Serialize model or encoder to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    """Load a pickle file and return the object."""
    with open(path, "rb") as f:
        return pickle.load(f)

def performance_on_categorical_slice(model, data, column_name, slice_value, encoder, lb):
    """Compute metrics on a slice of the data."""
    data_slice = data[data[column_name] == slice_value]
    X_slice, y_slice, _, _ = process_data(
        X=data_slice,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    return compute_model_metrics(y_slice, preds)
