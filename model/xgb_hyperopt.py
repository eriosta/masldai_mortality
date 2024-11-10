import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope


class XGBoostModelTrainer:
    def __init__(self, data_path, n_splits=3, use_hyperopt=False):
        self.data_path = data_path
        self.n_splits = n_splits
        self.use_hyperopt = use_hyperopt
        self.models = []
        self.performance_metrics = []
        self.random_state = 42

    def load_data(self):
        data = pd.read_csv(self.data_path).dropna(subset=['isF3'])
        data = data[
            [
                'isF3',
                'Age (years)',
                'Glycohemoglobin (%)',
                'Alanine aminotransferase (U/L)',
                'Aspartate aminotransferase (U/L)',
                'Platelet count (1000 cells/µL)',
                'Body-mass index (kg/m**2)',
                'GFR',
            ]
        ]
        X = data.drop(columns='isF3')
        y = data['isF3']
        return X, y

    def tune_hyperparameters(self, X, y):
        def objective(params):
            params['seed'] = self.random_state  # Ensure consistent random state
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            combined_scores = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                model = xgb.train(params, dtrain, num_boost_round=100)
                val_pred = model.predict(dval)
                val_pred_binary = (val_pred >= 0.5).astype(int)

                # Calculate specificity and PPV
                tn, fp, fn, tp = confusion_matrix(y_val, val_pred_binary).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

                # Combine specificity and PPV (maximize both)
                combined_score = (specificity + ppv) / 2
                combined_scores.append(combined_score)

            return -np.mean(combined_scores)  # Negative combined score to maximize it

        # Expanded hyperparameter search space
        space = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',  # Maximize AUROC
            'scale_pos_weight': len(y[y == 0]) / len(y[y == 1]),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
            'eta': hp.loguniform('eta', np.log(0.001), np.log(0.5)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 30, 1),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            'gamma': hp.uniform('gamma', 0, 20),
            'lambda': hp.uniform('lambda', 0.1, 10),
            'alpha': hp.uniform('alpha', 0, 10),
        }

        trials = Trials()
        rng = np.random.default_rng(self.random_state)
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=rng)
        return best_params

    def train_model(self):
        X, y = self.load_data()
        if self.use_hyperopt:
            params = self.tune_hyperparameters(X, y)
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
            params['scale_pos_weight'] = len(y[y == 0]) / len(y[y == 1])
            params['max_depth'] = int(params['max_depth'])
            params['min_child_weight'] = int(params['min_child_weight'])
            params['seed'] = self.random_state
        else:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': len(y[y == 0]) / len(y[y == 1]),
                'max_depth': 5,
                'eta': 0.1,
                'seed': self.random_state,
            }

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(params, dtrain, num_boost_round=100)

            val_pred = model.predict(dval)
            val_pred_binary = (val_pred >= 0.5).astype(int)

            auroc = roc_auc_score(y_val, val_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, val_pred_binary).ravel()
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            self.performance_metrics.append({
                'Fold': fold,
                'AUROC': auroc,
                'PPV': ppv,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'NPV': npv,
            })
            self.models.append(model)

        metrics_df = pd.DataFrame(self.performance_metrics)
        metrics_df.loc[len(metrics_df)] = metrics_df.mean().astype(float)
        metrics_df.at[len(metrics_df) - 1, 'Fold'] = 'Average'

        with open('xgboost_model_11_10_2024.pkl', 'wb') as file:
            pickle.dump(self.models[-1], file)
        return metrics_df


# Example usage
trainer = XGBoostModelTrainer(data_path='masld_f3_n_1373.csv', n_splits=5, use_hyperopt=True)
metrics_df = trainer.train_model()
metrics_df
metrics_df.to_csv('k_fold_performance_metrics.csv')