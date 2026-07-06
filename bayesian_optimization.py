from collections import Counter
import warnings

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, check_array, check_is_fitted, clone, is_classifier
from sklearn.model_selection import cross_val_score
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from sklearn.utils.validation import validate_data
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern

class BayesianOptimization(MetaEstimatorMixin, BaseEstimator):
    """
    Bayesian Optimization Model

    Parameters
    ----------
    estimator : BaseEstimator
        The base model to test hyper parameters against
    param_distributions : dict
        Parameter distributions for each hyper parameter.
        The values for each of the hyper parameters can be either:
        - **list** - list of categorical values, like kernels for example,
        - **discrete rvs** - such as `scipy.stats.randint`,
        - **continuous rvs** - such as `scipy.stats.loguniform`
    n_iter : int, default = 10
        The number of iterations to run when testing new hyper parameters
    n_initial : int, default = 5
        The initial number of hyper parameters to start with before iterating
    cv : int, default = 5
        The number of K-folds to use when scoring the hyper parameter with the estimator
    scoring : str, or None, default = None
        The type of scoring to use. If `None`, uses the best scoring method based on the estimator
    random_state : int, or RandomState, or None, default = None
    kernel : Kernel or None, default = Matern
        The type of kernel to use with Gaussian Process Regressor. 
        By default it is set to the **Matern** kernel. Setting it to **None** will use
        GaussianProcessRegressor's default kernel option.
    gp_alpha : float, default = 1e-10
        The alpha value passed into the GaussianProcessRegressor
    exploration : float, default = 0.01
        The exploration value threshold which determines how much the expected improvement
        of Bayesian optimization will lean towards exploration over exploitation.
    n_candidates : int, default = 10000
        The number of configs to randomly generate at each iteration when finding the next best
        approximate config using Gaussian Process.
    logging : bool, default = False
        Used for internal debugging and tracking of the estimator

    Examples
    --------
    ```python
>>> from sklearn.datasets import load_iris
>>> from scipy.stats import loguniform
>>> from sklearn.svm import SVC
>>> from bayesian_optimization import BayesianOptimization
>>> params = {
>>>     "C" : loguniform(1e-2, 1e+2),
>>>     "gamma" : loguniform(1e-4, 1)
>>> }
>>> X, y = load_iris(return_X_y=True)
>>> bo = BayesianOptimization(SVC(), params)
>>> bo.fit(X, y)
>>> print(bo.best_estimator_, bo.best_score_)
SVC(C=np.float64(15.244520952308381), gamma=np.float64(0.01749729594777597)) 0.9866666666666667
    ```
    """
    def __init__(
            self,
            estimator: BaseEstimator,
            param_distributions: dict[str, rv_continuous_frozen],
            *,
            n_iter: int = 10,
            n_initial: int = 5,
            cv: int | None = None,
            scoring: str | None = None,
            random_state: int | np.random.RandomState | None = None,
            kernel: Kernel | None = Matern(),
            gp_alpha: float = 1e-10,
            exploration: float = 0.01,
            n_candidates: int = 10_000,
            logging: bool = False):
        
        self.estimator = estimator        
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.n_initial = n_initial
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.kernel = kernel
        self.gp_alpha = gp_alpha
        self.exploration = exploration
        self.n_candidates = n_candidates
        self.logging = logging

    class Config:
        """Parameter configuration class"""
        def __init__(self, params: dict[str, np.float64]):
            self.params = params

        def __str__(self):
            return " ".join(
                f"{param}: {value:.4f}" 
                for param, value in self.params.items()
            )
        
    def _print(self, msg: str):
        """Internal logger for all printing"""
        if self.logging:
            print(msg)
        
    def _check_random_state(self) -> np.random.RandomState:
        """Make sure that the random state variable is valid"""
        if isinstance(self.random_state, np.random.RandomState):
            return self.random_state
        elif isinstance(self.random_state, int):
            return np.random.RandomState(self.random_state)
        else:
            return np.random.RandomState()
        
    def _gen_configs_from_param_distributions(self, no_configs: int) -> tuple[np.ndarray, list[Config]]:
        """Generate `no_configs` configurations to test"""
        param_names = list(self.param_distributions.keys())
        sampled = {
            param: dist.rvs(size=no_configs, random_state=self._rng)
            for param, dist in self.param_distributions.items()
        }

        configs = [
            self.Config(
                params={param: sampled[param][i] for param in param_names}
            )
            for i in range(no_configs)
        ]

        return np.array([
            [sampled[param][i] for param in param_names]
            for i in range(no_configs)
        ]), configs
        
    def _get_valid_n_splits(self, y) -> int:
        """
        Determine if the set CV (n_splits) value is adequate for the data subset.

        For classification, this is needed as cross validation can only allow a n_splits 
        value that is greater than or equal to the minimum number of members in each class.

        In the case of regression, we simply take the minimum between the set CV value and the
        length of the subset.
        """
        requested = self.cv if self.cv is not None else 5

        if is_classifier(self.estimator):
            min_class_count = min(Counter(y).values())  # Get the minimum number of members for each class
            if min_class_count < requested:
                warnings.warn(
                    "min_resources may be too small for the smallest class to be "
                    "represented in early rungs. Consider increasing min_resources."
                )
            n_splits = min(requested, min_class_count)
        else:
            n_splits = min(requested, len(y))

        return max(2, n_splits)
    
    def expected_improvement(self, candidates: np.ndarray) -> np.ndarray:
        """Get the expected improvements for each candidate."""
        mu, sigma = self._gp.predict(candidates, return_std=True)
        f_best = self._y_obs.min()

        with np.errstate(divide="ignore", invalid="ignore"):
            Z = (f_best - mu - self.exploration)/ (sigma)
            ei = (f_best - mu - self.exploration) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-10] = 0.0

        return ei
    
    def _next_point_by_ei(self) -> tuple[np.ndarray, Config]:
        """Get the next best point to add to the configurations for testing."""
        candidates, candidate_configs = self._gen_configs_from_param_distributions(self.n_candidates)
        ei_values = self.expected_improvement(candidates)

        max_ei_index = np.argmax(ei_values)

        return candidates[max_ei_index], candidate_configs[max_ei_index]
    
    def _evaluate(self, config: Config) -> np.float64:
        """
        Evaluate the model's loss for a single config using cross-validation.

        Parameters
        ----------
        config : BayesianOptimization.Config
                Config of yper parameters to test,

        Returns
        -------
        float
            Mean cross-validated loss for the config
        """
        estimator = clone(self.estimator).set_params(**config.params)

        scores = cross_val_score(
            estimator,
            self.X_, self.y_,
            cv=self._n_splits,
            scoring=self.scoring
        )

        return 1 - scores.mean()
        
    def _validate_params(self):
        """Validate the parameters of the estimator before fitting it with the data"""
        for key, dist in self.param_distributions.items():
            if not isinstance(dist, rv_continuous_frozen):
                raise TypeError(f"{key} is not a scipy.stats.rv_continuous_frozen type!")

    def fit(self, X, y):
        self._print("Bayesian Optimization Starting...")

        self._validate_params()

        self._rng = self._check_random_state()

        X, y = validate_data(self, X, y)

        self.X_, self.y_ = X, y

        self._n_splits = self._get_valid_n_splits(self.y_)

        self._gp = GaussianProcessRegressor(
            self.kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            alpha=self.gp_alpha,
            random_state=self._rng
        )

        self._X_obs, configs = self._gen_configs_from_param_distributions(self.n_initial)
    
        self._y_obs = np.array([
            self._evaluate(config)
            for config in configs
        ])

        for i in range(self.n_iter):
            self._print(f"Iteration {i}: GP fitting...")
            self._gp.fit(self._X_obs, self._y_obs)

            self._print(f"Iteration {i}: Getting next point...")
            next_x, next_config = self._next_point_by_ei()

            self._print(f"Iteration {i}: Evaluating best config...")
            next_y = self._evaluate(next_config)

            self._X_obs = np.vstack([self._X_obs, next_x])
            configs.append(next_config)
            self._y_obs = np.append(self._y_obs, next_y)
            self._print(f"Iteration {i + 1} complete")

        best_idx = np.argmin(self._y_obs)
        best_config = configs[best_idx]

        self.best_params_ = best_config.params
        self.best_score_ = 1 - self._y_obs[best_idx]

        self._print("Best config found:")
        self._print("==================")
        self._print(f"{best_config}")
        self._print(f"{f"{self.scoring} score" if self.scoring is not None else "Score"}: {self.best_score_}")

        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(self.X_, self.y_)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """Predict the outputs `y` based on inputs `X`"""
        check_is_fitted(self)
        X = check_array(X)
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """Score the best estimator on the given data"""
        check_is_fitted(self)
        X, y = validate_data(self, X, y)
        return self.best_estimator_.score(X, y)
    
    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value"""
        return hasattr(self, "_is_fitted") and self._is_fitted