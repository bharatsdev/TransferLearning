import pickle

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from dogs_vs_cats.config import dog_vs_cat_config as config

db = h5py.File(config.EXTRACT_FEATURES_HDF5, 'r')
idx = int(db['labels'].shape[0] * 75)

print('[INFO] : Hyperparameters tunnings...!')
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(solver='lbfgs', multi_class='auto'), params, cv=3, n_jobs=-1)

model.fit(db['features'][:idx], db['labels'][:idx])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

print('[INFO] : Evaluting the model....Q')
preds = model.predict(db['featues'][idx:])
print(classification_report(db['features'][idx:], preds, target_names=db['label_names']))

acc = accuracy_score(db["labels"][idx:], preds)
print("[INFO] score: {}".format(acc))

print('[INFO] : Saving model...!')
with open(config.MODEL_PATH, 'wb') as f:
    f.write(pickle.dumps(model.best_estimator_))
