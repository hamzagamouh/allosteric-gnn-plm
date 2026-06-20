from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef


def train_rf(train_feats, y_train, val_feats, y_val,
             n_estimators=10, n_jobs=-1, random_state=454356):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
    )
    rf.fit(train_feats, y_train.ravel())
    train_mcc = matthews_corrcoef(y_train.ravel(), rf.predict(train_feats))
    val_mcc = matthews_corrcoef(y_val.ravel(), rf.predict(val_feats))
    return train_mcc, val_mcc, rf.feature_importances_
