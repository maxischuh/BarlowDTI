import pandas as pd
import numpy as np
from pathlib import Path, PurePath

from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, accuracy_score, f1_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier, XGBRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor

import optuna
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, InCondition
from smac import Scenario, MultiFidelityFacade


class XGBCls:
    """ XGBoost Classifier with SMAC3 hyperparameter optimization"""
    def __init__(
            self,
            seed: int = 42,
            max_n_estimators: int = 1000,
            eval_metric: str = None
    ):
        self.optimized = False
        self.max_n_estimators = max_n_estimators
        self.best_params = None
        self.seed = seed
        self.cls = None
        self.eval_metric = eval_metric
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

        """
        self.space = ConfigurationSpace(
            name="XGB-hspace",
            seed=self.seed,
            space={
                "learning_rate": Float("learning_rate", (1e-8, 1.0), log=True),
                "max_depth": Integer("max_depth", (2, 12)),
                "gamma": Float("gamma", (1e-8, 1.0), log=True),
                "min_child_weight": Float("min_child_weight", (1e-8, 1e2), log=True),
                "subsample": Float("subsample", (0.4, 1.0)),
                # "colsample_bytree": Float("colsample_bytree", (0.2, 1.0)),
                "reg_lambda": Float("reg_lambda", (1e-6, 10), log=True),

                "objective": Categorical("objective", ["binary:logistic", "focal_loss", "binary:hinge"]),
                "loss_gamma": Float("loss_gamma", (1.0, 5.0)),
                "loss_epsilon": Float("loss_epsilon", (0.0, 5.0)),
            }
        )
        
        """
        self._gamma = None
        self._epsilon = None

        # Define the space without the loss_gamma and loss_epsilon hyperparameters
        self.space = ConfigurationSpace(name="XGB-hspace", seed=self.seed)
        learning_rate = Float("learning_rate", (1e-8, 1.0), log=True)
        max_depth = Integer("max_depth", (2, 12))
        gamma = Float("gamma", (1e-8, 1.0), log=True)
        min_child_weight = Float("min_child_weight", (1e-8, 1e2), log=True)
        subsample = Float("subsample", (0.4, 1.0))
        reg_lambda = Float("reg_lambda", (1e-6, 10), log=True)

        objective = Categorical("objective", ["binary:logistic", "focal_loss"])
        loss_gamma = Float("loss_gamma", (1.0, 5.0))
        loss_epsilon = Float("loss_epsilon", (0.0, 5.0))

        self.space.add_hyperparameters([learning_rate, max_depth, gamma, min_child_weight, subsample, reg_lambda, objective, loss_gamma, loss_epsilon])

        cond_loss_gamma = InCondition(child=loss_gamma, parent=objective, values=["focal_loss"])
        cond_loss_epsilon = InCondition(child=loss_epsilon, parent=objective, values=["focal_loss"])
        self.space.add_conditions([cond_loss_gamma, cond_loss_epsilon])

    def _focal_loss(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            # gamma: float = 2.0,
            # epsilon: float = 0.4,
            class_weight: str = "balanced",
            e: float = 1e-3
    ):
        # Define the class weights
        if class_weight == "balanced":
            n_majority = len(y_true) - np.sum(y_true)
            n_minority = np.sum(y_true)

            majority_ratio = 1 - (n_majority / (n_majority + n_minority))
            minority_ratio = 1 - (n_minority / (n_majority + n_minority))
            alpha_m = minority_ratio
            alpha_M = majority_ratio
        elif class_weight is None:
            alpha_m = 1.0
            alpha_M = 1.0
        else:
            alpha_m = class_weight[0]
            alpha_M = class_weight[1]

        def call(y_true, y_pred):
            p = 1 / (1 + np.exp(-y_pred))
            q = 1 - p

            focal_pos = np.power(q, self._gamma) * np.log(p)
            poly1_pos = self._epsilon * np.power(q, self._gamma + 1.0)
            pos_loss = np.add(focal_pos, poly1_pos) * alpha_m

            focal_neg = np.power(p, self._gamma) * np.log(q)
            poly1_neg = self._epsilon * np.power(p, self._gamma + 1.0)
            neg_loss = np.add(focal_neg, poly1_neg) * alpha_M

            return y_true * pos_loss + (1 - y_true) * neg_loss

        # Calculate the gradients and Hessians
        loss = call(y_true, y_pred)
        diff1 = call(y_true, y_pred + e)
        diff2 = call(y_true, y_pred - e)

        grad = - (diff1 - diff2) / (2 * e)
        hess = - (diff1 - 2 * loss + diff2) / e ** 2

        return grad, hess

    def _train(self, config: Configuration, seed: int, budget: int):
        config = dict(config)

        if config["objective"] == "focal_loss":
            self._gamma = config.pop("loss_gamma")
            self._epsilon = config.pop("loss_epsilon")
            obj = self._focal_loss
        else:
            obj = config["objective"]

        config.pop("objective")

        opt_cls = XGBClassifier(
            n_estimators=int(np.ceil(budget)),
            random_state=seed,
            n_jobs=-1,
            objective=obj,
            # eval_metric=roc_pr_min,
            # early_stopping_rounds=int(np.ceil(budget/5)),
            **config
        )
        opt_cls.fit(self.x_train, self.y_train)
        pred = opt_cls.predict_proba(self.x_valid)[:, 1]
        roc, pr = self.metric(self.y_valid, pred)

        if self.eval_metric == "roc":
            return 1 - roc
        elif self.eval_metric == "pr":
            return 1 - pr
        else:
            return 2 - pr - roc

    def optimize(self, x_train, y_train, x_valid, y_valid, n_trials: int = 200, name: str = "train"):
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid

        scenario = Scenario(
            configspace=self.space,
            name=f"XGB-{name}-{n_trials}-smac3",
            n_trials=n_trials,
            min_budget=100,
            max_budget=self.max_n_estimators,
            seed=self.seed,
        )

        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=self._train,
        )

        self.optimized = True
        self.best_params = dict(smac.optimize())

    def fit(self, x, y, x_valid=None, y_valid=None, **kwargs):
        if self.optimized:
            exec_best_params = self.best_params.copy()

            if exec_best_params["objective"] == "focal_loss":
                self._gamma = exec_best_params.pop("loss_gamma")
                self._epsilon = exec_best_params.pop("loss_epsilon")
                obj = self._focal_loss
            else:
                obj = exec_best_params["objective"]

            exec_best_params.pop("objective")

            self.cls = XGBClassifier(
                n_estimators=self.max_n_estimators,
                random_state=self.seed,
                n_jobs=-1,
                # eval_metric=roc_pr_min,
                # early_stopping_rounds=int(np.ceil(self.max_n_estimators/5)),
                objective=obj,
                **exec_best_params
            )
        else:
            self.cls = XGBClassifier(
                random_state=self.seed,
                n_jobs=-1,
                **kwargs
            )

        self.cls.fit(x, y)  # eval_set=[(x_valid, y_valid)]

    def predict(self, x):
        return self.cls.predict(x)

    def predict_proba(self, x):
        return self.cls.predict_proba(x)

    @staticmethod
    def metric(y_true, y_pred):
        try:
            roc = roc_auc_score(y_true, y_pred)
            pr = average_precision_score(y_true, y_pred)
        except ValueError:
            roc, pr = None, None
        return roc, pr

    def save(self, path):
        # Create directory if not exists from variable path
        Path(path).mkdir(parents=True, exist_ok=True)
        # Save model
        self.cls.save_model(PurePath(path, "model.xgb"))
        if self.optimized:
            pd.DataFrame(self.best_params, index=[0]).to_csv(Path(path, "best_params.csv"))

    def load(self, path):
        self.cls = XGBClassifier()
        self.cls.load_model(str(Path(path, "model.xgb")))
        # Load if file exists
        if Path(path, "best_params.csv").is_file():
            self.best_params = pd.read_csv(PurePath(path, "best_params.csv"), index_col=0).to_dict(orient="records")[0]
            self.optimized = True
        else:
            self.optimized = False


class XGBClsOptuna:
    """ XGBoost Classifier with Optuna hyperparameter optimization"""
    def __init__(
            self,
            seed: int = 42,
            max_n_estimators: int = 1000,
            n_jobs: int = -1,
            eval_metric: str = None
    ):
        self.optimized = False
        self.max_n_estimators = max_n_estimators
        self.n_jobs = n_jobs
        self.best_params = None
        self.seed = seed
        self.cls = None
        self.study = None
        self.eval_metric = eval_metric
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self._gamma = None
        self._epsilon = None

    def _focal_loss(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            # gamma: float = 2.0,
            # epsilon: float = 0.4,
            class_weight: str = "balanced",
            e: float = 1e-3
    ):
        # Define the class weights
        if class_weight == "balanced":
            n_majority = len(y_true) - np.sum(y_true)
            n_minority = np.sum(y_true)

            majority_ratio = 1 - (n_majority / (n_majority + n_minority))
            minority_ratio = 1 - (n_minority / (n_majority + n_minority))
            alpha_m = minority_ratio
            alpha_M = majority_ratio
        elif class_weight is None:
            alpha_m = 1.0
            alpha_M = 1.0
        else:
            alpha_m = class_weight[0]
            alpha_M = class_weight[1]

        def call(y_true, y_pred):
            p = 1 / (1 + np.exp(-y_pred))
            q = 1 - p

            focal_pos = np.power(q, self._gamma) * np.log(p)
            poly1_pos = self._epsilon * np.power(q, self._gamma + 1.0)
            pos_loss = np.add(focal_pos, poly1_pos) * alpha_m

            focal_neg = np.power(p, self._gamma) * np.log(q)
            poly1_neg = self._epsilon * np.power(p, self._gamma + 1.0)
            neg_loss = np.add(focal_neg, poly1_neg) * alpha_M

            return y_true * pos_loss + (1 - y_true) * neg_loss

        # Calculate the gradients and Hessians
        loss = call(y_true, y_pred)
        diff1 = call(y_true, y_pred + e)
        diff2 = call(y_true, y_pred - e)

        grad = - (diff1 - diff2) / (2 * e)
        hess = - (diff1 - 2 * loss + diff2) / e ** 2

        return grad, hess

    def _train(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 100, self.max_n_estimators, step=100)
        learning_rate = trial.suggest_float("learning_rate", 1e-8, 1.0, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 12)
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        min_child_weight = trial.suggest_float("min_child_weight", 1e-8, 1e2, log=True)
        subsample = trial.suggest_float("subsample", 0.4, 1.0)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 10, log=True)

        """

        objective = trial.suggest_categorical("objective", ["binary:logistic", "focal_loss"])

        if objective == "focal_loss":
            loss_gamma = trial.suggest_float("loss_gamma", 1.0, 5.0)
            loss_epsilon = trial.suggest_float("loss_epsilon", 0.0, 5.0)
            self._gamma = loss_gamma
            self._epsilon = loss_epsilon
            objective = self._focal_loss
        """

        opt_cls = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            reg_lambda=reg_lambda,
            # objective=objective,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        opt_cls.fit(self.x_train, self.y_train)

        if self.eval_metric == "roc" or self.eval_metric == "pr":
            pred = opt_cls.predict_proba(self.x_valid)[:, 1]

            try:
                roc = roc_auc_score(self.y_valid, pred)
                pr = average_precision_score(self.y_valid, pred)
            except ValueError:
                roc, pr = None, None
    
            if self.eval_metric == "roc":
                return roc
            elif self.eval_metric == "pr":
                return pr
            else:
                return pr + roc
            
        elif self.eval_metric == "acc":
            pred = opt_cls.predict(self.x_valid)
            acc = accuracy_score(self.y_valid, pred)
            return acc
        
        elif self.eval_metric == "f1":
            pred = opt_cls.predict(self.x_valid)
            f1 = f1_score(self.y_valid, pred, average="macro")
            return f1

    def optimize(self, x_train, y_train, x_valid, y_valid, n_trials: int = 100, name: str = "train"):
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid

        self.study = optuna.create_study(direction="maximize", study_name=f"XGB-{name}-{n_trials}-optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(self._train, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

        self.optimized = True
        self.best_params = dict(self.study.best_trial.params)

    def fit(self, x, y, x_valid=None, y_valid=None, **kwargs):
        if self.optimized:
            exec_best_params = self.best_params.copy()

            if exec_best_params["objective"] == "focal_loss":
                self._gamma = exec_best_params.pop("loss_gamma")
                self._epsilon = exec_best_params.pop("loss_epsilon")
                obj = self._focal_loss
            else:
                obj = exec_best_params["objective"]

            exec_best_params.pop("objective")

            self.cls = XGBClassifier(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                # eval_metric=roc_pr_min,
                # early_stopping_rounds=int(np.ceil(self.max_n_estimators/5)),
                objective=obj,
                **exec_best_params
            )
        else:
            self.cls = XGBClassifier(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                **kwargs
            )

        self.cls.fit(x, y)  # eval_set=[(x_valid, y_valid)]

    def predict(self, x):
        return self.cls.predict(x)

    def predict_proba(self, x):
        return self.cls.predict_proba(x)

    def save(self, path):
        # Create directory if not exists from variable path
        Path(path).mkdir(parents=True, exist_ok=True)
        # Save model
        self.cls.save_model(PurePath(path, "model.xgb"))
        if self.optimized:
            self.study.trials_dataframe().to_csv(str(PurePath(path, "best_params_optuna.csv")), index=False)
            optuna.visualization.plot_optimization_history(self.study).write_html(str(PurePath(path, "optimization_history.html")))
            optuna.visualization.plot_param_importances(self.study).write_html(str(PurePath(path, "param_importances.html")))

    def load(self, path):
        self.cls = XGBClassifier()
        self.cls.load_model(str(Path(path, "model.xgb")))
        # Load if file exists
        if Path(path, "best_params.csv").is_file():
            self.best_params = pd.read_csv(PurePath(path, "best_params_optuna.csv"), index_col=0).to_dict(orient="records")[0]
            self.optimized = True
        else:
            self.optimized = False


class XGBRegOptuna:
    """ XGBoost Regressor with Optuna hyperparameter optimization"""
    def __init__(
            self,
            seed: int = 42,
            max_n_estimators: int = 1000,
            eval_metric: str = None
    ):
        self.optimized = False
        self.max_n_estimators = max_n_estimators
        self.best_params = None
        self.seed = seed
        self.reg = None
        self.study = None
        self.eval_metric = eval_metric
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self._gamma = None
        self._epsilon = None

    def _focal_loss(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            # gamma: float = 2.0,
            # epsilon: float = 0.4,
            class_weight: str = "balanced",
            e: float = 1e-3
    ):
        # Define the class weights
        if class_weight == "balanced":
            n_majority = len(y_true) - np.sum(y_true)
            n_minority = np.sum(y_true)

            majority_ratio = 1 - (n_majority / (n_majority + n_minority))
            minority_ratio = 1 - (n_minority / (n_majority + n_minority))
            alpha_m = minority_ratio
            alpha_M = majority_ratio
        elif class_weight is None:
            alpha_m = 1.0
            alpha_M = 1.0
        else:
            alpha_m = class_weight[0]
            alpha_M = class_weight[1]

        def call(y_true, y_pred):
            p = 1 / (1 + np.exp(-y_pred))
            q = 1 - p

            focal_pos = np.power(q, self._gamma) * np.log(p)
            poly1_pos = self._epsilon * np.power(q, self._gamma + 1.0)
            pos_loss = np.add(focal_pos, poly1_pos) * alpha_m

            focal_neg = np.power(p, self._gamma) * np.log(q)
            poly1_neg = self._epsilon * np.power(p, self._gamma + 1.0)
            neg_loss = np.add(focal_neg, poly1_neg) * alpha_M

            return y_true * pos_loss + (1 - y_true) * neg_loss

        # Calculate the gradients and Hessians
        loss = call(y_true, y_pred)
        diff1 = call(y_true, y_pred + e)
        diff2 = call(y_true, y_pred - e)

        grad = - (diff1 - diff2) / (2 * e)
        hess = - (diff1 - 2 * loss + diff2) / e ** 2

        return grad, hess

    def _train(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 100, self.max_n_estimators, step=100)
        learning_rate = trial.suggest_float("learning_rate", 1e-8, 1.0, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 12)
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        min_child_weight = trial.suggest_float("min_child_weight", 1e-8, 1e2, log=True)
        subsample = trial.suggest_float("subsample", 0.4, 1.0)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 10, log=True)

        """

        objective = trial.suggest_categorical("objective", ["binary:logistic", "focal_loss"])

        if objective == "focal_loss":
            loss_gamma = trial.suggest_float("loss_gamma", 1.0, 5.0)
            loss_epsilon = trial.suggest_float("loss_epsilon", 0.0, 5.0)
            self._gamma = loss_gamma
            self._epsilon = loss_epsilon
            objective = self._focal_loss
        """

        opt_reg = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            reg_lambda=reg_lambda,
            # objective=objective,
            random_state=self.seed,
            n_jobs=-1,
        )
        opt_reg.fit(self.x_train, self.y_train)
        pred = opt_reg.predict(self.x_valid)
        mae, pcc = self.metric(self.y_valid, pred)

        if self.eval_metric == "mae":
            return mae
        elif self.eval_metric == "pcc":
            return (-1 * pcc)
        else:
            return mae + (-1 * pcc)

    def optimize(self, x_train, y_train, x_valid, y_valid, n_trials: int = 100, name: str = "train"):
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid

        self.study = optuna.create_study(direction="minimize", study_name=f"XGB-{name}-{n_trials}-optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(self._train, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

        self.optimized = True
        self.best_params = dict(self.study.best_trial.params)

    def fit(self, x, y, x_valid=None, y_valid=None, **kwargs):
        if self.optimized:
            exec_best_params = self.best_params.copy()

            # if exec_best_params["objective"] == "focal_loss":
            #     self._gamma = exec_best_params.pop("loss_gamma")
            #     self._epsilon = exec_best_params.pop("loss_epsilon")
            #     obj = self._focal_loss
            # else:
            #     obj = exec_best_params["objective"]

            # exec_best_params.pop("objective")

            self.reg = XGBRegressor(
                random_state=self.seed,
                n_jobs=-1,
                # eval_metric=roc_pr_min,
                # early_stopping_rounds=int(np.ceil(self.max_n_estimators/5)),
                # objective=obj,
                **exec_best_params
            )
        else:
            self.reg = XGBRegressor(
                random_state=self.seed,
                n_jobs=-1,
                **kwargs
            )

        self.reg.fit(x, y)  # eval_set=[(x_valid, y_valid)]

    def predict(self, x):
        return self.reg.predict(x)

    @staticmethod
    def metric(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        pcc = pearsonr(y_true, y_pred).statistic
        print(f"MAE: {mae}, PCC: {pcc}")
        return mae, pcc

    def save(self, path):
        # Create directory if not exists from variable path
        Path(path).mkdir(parents=True, exist_ok=True)
        # Save model
        self.reg.save_model(PurePath(path, "reg_model.xgb"))
        if self.optimized:
            self.study.trials_dataframe().to_csv(str(PurePath(path, "reg_best_params_optuna.csv")), index=False)
            optuna.visualization.plot_optimization_history(self.study).write_html(str(PurePath(path, "reg_optimization_history.html")))
            optuna.visualization.plot_param_importances(self.study).write_html(str(PurePath(path, "reg_param_importances.html")))

    def load(self, path):
        self.reg = XGBRegressor()
        self.reg.load_model(str(Path(path, "reg_reg_model.xgb")))
        # Load if file exists
        if Path(path, "reg_best_params.csv").is_file():
            self.best_params = pd.read_csv(PurePath(path, "reg_best_params_optuna.csv"), index_col=0).to_dict(orient="records")[0]
            self.optimized = True
        else:
            self.optimized = False
