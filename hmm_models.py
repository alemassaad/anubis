# hmm_models.py

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import multivariate_normal

class MultivariateGaussian:
    """
    Stores mean and covariance for a Gaussian distribution. Provides a PDF method.
    """
    def __init__(self, means, cov):
        self.means = means
        self.cov = cov

    def pdf(self, x):
        return multivariate_normal(mean=self.means, cov=self.cov).pdf(x)


class GaussianMixture:
    """
    Stores multiple (weight, MultivariateGaussian) pairs for mixture modeling.
    Provides a PDF by summing component PDFs weighted by mixture weights.
    """
    def __init__(self, weights, gaussians):
        self.weights = weights
        self.gaussians = gaussians

    def pdf(self, x):
        total = 0.0
        for w, g in zip(self.weights, self.gaussians):
            total += w * g.pdf(x)
        return total


class GaussianHMMWrapper:
    """
    Wraps hmmlearn's GaussianHMM. Provides parameter counting methods and 
    a transform method for state prediction. 
    """
    def __init__(self, n_regimes, covariance_type='full', n_iter=100, tol=1e-3):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        from hmmlearn.hmm import GaussianHMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol
        )
        self.regime_list = [None] * n_regimes
        self.regime_dict = {}
        self.index_remap = {i: i for i in range(n_regimes)}

    def fit(self, data: pd.DataFrame):
        self.model.fit(data)
        for i in range(self.n_regimes):
            mg = MultivariateGaussian(
                means=self.model.means_[i],
                cov=self.model.covars_[i]
            )
            self.regime_list[i] = mg
        sorted_idx = np.argsort(self.model.means_[:, 0])
        self.index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}
        self.regime_list = [self.regime_list[i] for i in sorted_idx]
        self.regime_dict = {f"regime{i+1}": r for i, r in enumerate(self.regime_list)}
        return self

    def transform(self, data: pd.DataFrame) -> pd.Series:
        hidden_states = self.model.predict(data)
        mapped = pd.Series(hidden_states, index=data.index).map(self.index_remap)
        return mapped.apply(lambda x: f"regime{x+1}")

    def score(self, data: pd.DataFrame) -> float:
        return self.model.score(data)

    def num_params(self) -> int:
        """
        Returns the count of free parameters for a GaussianHMM.
        """
        n_components = self.model.n_components
        n_features = self.model.means_.shape[1]
        init_params = n_components - 1
        trans_params = n_components * (n_components - 1)

        if self.covariance_type == 'full':
            cov_params_per_state = (n_features * (n_features + 1)) // 2
        elif self.covariance_type == 'diag':
            cov_params_per_state = n_features
        else:
            raise NotImplementedError("Unsupported covariance type.")

        emission_means = n_components * n_features
        emission_covs = n_components * cov_params_per_state

        return init_params + trans_params + emission_means + emission_covs

    def num_params_empirical(self) -> int:
        """
        Returns a parameter count by inspecting array shapes in the fitted model.
        For debugging or sanity check.
        """
        n_components = self.model.n_components
        init_params = n_components - 1
        trans_params = n_components * (n_components - 1)

        means_params = self.model.means_.size

        if self.covariance_type == 'full':
            cov_params = 0
            for c in self.model.covars_:
                n_features = c.shape[0]
                cov_params += (n_features*(n_features+1))//2
        elif self.covariance_type == 'diag':
            cov_params = self.model.covars_.size
        else:
            raise NotImplementedError("Unsupported covariance type.")

        return init_params + trans_params + means_params + cov_params


class GMMHMMWrapper:
    """
    Wraps hmmlearn's GMMHMM. Provides parameter counting methods and a transform method for state prediction.
    """
    def __init__(self, n_regimes, n_mix=2, covariance_type='full', n_iter=100, tol=1e-3):
        self.n_regimes = n_regimes
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        from hmmlearn.hmm import GMMHMM
        self.model = GMMHMM(
            n_components=self.n_regimes,
            n_mix=self.n_mix,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol
        )
        self.index_remap = None
        self.regime_list = None
        self.regime_dict = None

    def fit(self, data: pd.DataFrame):
        self.model.fit(data)
        mean_first_dim = self.model.means_[:, :, 0].mean(axis=1)
        sorted_idx = np.argsort(mean_first_dim)
        self.index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}

        regime_list = []
        for i in range(self.n_regimes):
            weights_i = self.model.weights_[i]
            gaussians_i = []
            for j in range(self.n_mix):
                mg = MultivariateGaussian(
                    means=self.model.means_[i, j],
                    cov=self.model.covars_[i, j]
                )
                gaussians_i.append(mg)
            mixture = GaussianMixture(weights=weights_i, gaussians=gaussians_i)
            regime_list.append(mixture)

        # reorder
        regime_list = [regime_list[i] for i in sorted_idx]
        self.regime_list = regime_list
        self.regime_dict = {f"regime{i+1}": r for i, r in enumerate(self.regime_list)}
        return self

    def transform(self, data: pd.DataFrame) -> pd.Series:
        hidden_states = self.model.predict(data)
        mapped = pd.Series(hidden_states, index=data.index).map(self.index_remap)
        return mapped.apply(lambda x: f"regime{x+1}")

    def score(self, data: pd.DataFrame) -> float:
        return self.model.score(data)

    def num_params(self) -> int:
        n_components = self.model.n_components
        n_mix = self.model.n_mix
        n_features = self.model.means_.shape[2]
        init_params = n_components - 1
        trans_params = n_components * (n_components - 1)

        if self.covariance_type == 'full':
            cov_params_per_component = (n_features * (n_features + 1)) // 2
        elif self.covariance_type == 'diag':
            cov_params_per_component = n_features
        else:
            raise NotImplementedError("Unsupported covariance type.")

        mixture_weights = n_components * (n_mix - 1)
        emission_per_state = n_mix * (n_features + cov_params_per_component)
        emission_total = n_components * emission_per_state
        return init_params + trans_params + mixture_weights + emission_total

    def num_params_empirical(self) -> int:
        n_components = self.model.n_components
        n_mix = self.model.n_mix
        init_params = n_components - 1
        trans_params = n_components * (n_components - 1)
        mixture_weights = n_components * (n_mix - 1)
        means_params = self.model.means_.size

        if self.covariance_type == 'full':
            cov_params = 0
            for state_covar in self.model.covars_:
                for c in state_covar:
                    n_features = c.shape[0]
                    cov_params += (n_features*(n_features+1))//2
        elif self.covariance_type == 'diag':
            cov_params = self.model.covars_.size
        else:
            raise NotImplementedError("Unsupported covariance type.")

        return init_params + trans_params + mixture_weights + means_params + cov_params











class GaussianARHMM:
    """
    A robust AR(1) HMM. Single-lag, with alpha optionally learned.
    Uses forward-backward to compute exact pairwise posteriors (xi),
    applies probability floors for transitions / startprob to avoid log(0) issues,
    and does multi-pass updates for alpha and AR coefficients in each M-step.
    """

    def __init__(
        self,
        n_regimes: int,
        covariance_type: str = 'full',
        n_iter: int = 100,
        tol: float = 1e-3,
        alpha: float = 1.0,
        alpha_is_free: bool = False,
        p: int = 1,
        floor_prob: float = 1e-12,   # floor to avoid zero probabilities
        max_alpha_passes: int = 2,   # times to re-update alpha & AR in each M-step
    ):
        """
        Parameters
        ----------
        n_regimes : int
            Number of hidden states (regimes).
        covariance_type : str
            'full' or 'diag'.
        n_iter : int
            Max EM iterations.
        tol : float
            Convergence threshold on log-likelihood.
        alpha : float
            Initial alpha for AR(1): X(t) ~ means[j] + alpha * A_j (X(t-1) - means[j]) + noise.
        alpha_is_free : bool
            If True, we reestimate alpha each M-step.
        p : int
            AR order (currently only p=1 implemented).
        floor_prob : float
            Minimal probability for transitions / startprobs to avoid log(0).
        max_alpha_passes : int
            Times we refine alpha and AR in the M-step mini-loop for better accuracy.
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.alpha = alpha
        self.alpha_is_free = alpha_is_free
        self.p = p  # only 1 used
        self.floor_prob = floor_prob
        self.max_alpha_passes = max_alpha_passes

        self.means_ = None
        self.covars_ = None
        self.ar_coeffs_ = None
        self.transmat_ = None
        self.startprob_ = None
        self.fitted_ = False

    ########################################################################
    # Initialization
    ########################################################################
    def _init_params(self, data: np.ndarray):
        n_features = data.shape[1]

        # means: shape (n_regimes, n_features)
        self.means_ = np.random.randn(self.n_regimes, n_features)
        # AR coeffs: shape (n_regimes, n_features, n_features)
        self.ar_coeffs_ = np.random.randn(self.n_regimes, n_features, n_features)

        # Covariances
        if self.covariance_type == 'full':
            self.covars_ = np.stack([np.eye(n_features) for _ in range(self.n_regimes)])
        elif self.covariance_type == 'diag':
            self.covars_ = np.ones((self.n_regimes, n_features))
        else:
            raise NotImplementedError(f"Unsupported covariance type {self.covariance_type}.")

        # Start prob, transmat
        self.startprob_ = np.full(self.n_regimes, 1.0 / self.n_regimes)
        self.transmat_ = np.full((self.n_regimes, self.n_regimes), 1.0 / self.n_regimes)

    ########################################################################
    # E-step: log-likelihood, forward-backward, gamma, xi
    ########################################################################
    def _compute_log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """
        log_likelihood[t, j] = log p(X(t) | state=j).
        For t=0, unconditional on X(t-1).
        For t>0, means_[j] + alpha*(ar_coeffs_[j] @ X(t-1) - means_[j]).
        """
        n_samples, d = data.shape
        log_likelihood = np.zeros((n_samples, self.n_regimes))

        # t=0
        for j in range(self.n_regimes):
            if self.covariance_type == 'full':
                log_likelihood[0, j] = _log_gaussian(data[0], self.means_[j], self.covars_[j])
            else:
                log_likelihood[0, j] = _log_gaussian_diag(data[0], self.means_[j], self.covars_[j])

        # t >= 1
        for t in range(1, n_samples):
            for j in range(self.n_regimes):
                ar_mean = self.means_[j] + self.alpha * (self.ar_coeffs_[j] @ (data[t-1] - self.means_[j]))
                if self.covariance_type == 'full':
                    log_likelihood[t, j] = _log_gaussian(data[t], ar_mean, self.covars_[j])
                else:
                    log_likelihood[t, j] = _log_gaussian_diag(data[t], ar_mean, self.covars_[j])

        return log_likelihood

    def _forward_pass(self, log_likelihood: np.ndarray) -> np.ndarray:
        """
        forward[t, j] = log p(x(0..t), state(t)=j).
        """
        n_samples, n_states = log_likelihood.shape
        forward = np.full((n_samples, n_states), -np.inf)

        # init
        for j in range(n_states):
            forward[0, j] = np.log(self.startprob_[j]) + log_likelihood[0, j]

        # recursion
        for t in range(1, n_samples):
            for j in range(n_states):
                temp = forward[t-1] + np.log(self.transmat_[:, j])
                forward[t, j] = np.logaddexp.reduce(temp) + log_likelihood[t, j]

        return forward

    def _backward_pass(self, log_likelihood: np.ndarray) -> np.ndarray:
        """
        backward[t, i] = log p(x(t+1..end) | state(t)=i).
        """
        n_samples, n_states = log_likelihood.shape
        backward = np.full((n_samples, n_states), -np.inf)

        # init
        backward[-1] = 0.0  # log(1)

        # recursion
        for t in range(n_samples - 2, -1, -1):
            for i in range(n_states):
                temp = np.log(self.transmat_[i]) + log_likelihood[t+1] + backward[t+1]
                backward[t, i] = np.logaddexp.reduce(temp)

        return backward

    def _compute_posteriors(self, forward: np.ndarray, backward: np.ndarray, log_likelihood: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        gamma[t, i] = p(state(t)=i | all data)
        xi[t, i, j] = p(state(t)=i, state(t+1)=j | all data)
        """
        n_samples, n_states = forward.shape
        ll = np.logaddexp.reduce(forward[-1])  # log p(data)
        gamma = np.exp(forward + backward - ll)

        # xi: shape (n_samples-1, n_states, n_states)
        xi = np.zeros((n_samples - 1, n_states, n_states))
        for t in range(n_samples - 1):
            temp = (forward[t, :, None] + np.log(self.transmat_) +
                    backward[t+1, None, :] + np.expand_dims(log_likelihood[t+1], axis=0))
            # temp shape: (n_states, n_states)
            # Now exponentiate and normalize
            max_val = np.max(temp)
            xi_t = np.exp(temp - max_val)
            xi_sum = np.sum(xi_t)
            xi[t] = xi_t / (xi_sum + 1e-300)

        return gamma, xi, ll

    ########################################################################
    # M-step
    ########################################################################
    def _update_parameters(self, data: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        n_samples, d = data.shape

        # (1) Start prob
        start = gamma[0]
        start += self.floor_prob
        start /= np.sum(start)
        self.startprob_ = start

        # (2) Transmat
        # sum over t of xi[t, i, j], then row-normalize with floors
        trans_counts = np.sum(xi, axis=0)  # shape (n_states, n_states)
        for i in range(self.n_regimes):
            row = trans_counts[i] + self.floor_prob
            row_sum = np.sum(row)
            self.transmat_[i] = row / row_sum

        # (3) Means
        # means_j = sum_t gamma[t,j]* X(t) / sum_t gamma[t,j]
        sums_ = np.zeros((self.n_regimes, d))
        counts_ = np.zeros(self.n_regimes)
        for t in range(n_samples):
            for j in range(self.n_regimes):
                w = gamma[t, j]
                sums_[j] += w * data[t]
                counts_[j] += w
        for j in range(self.n_regimes):
            denom = counts_[j] + 1e-300
            self.means_[j] = sums_[j] / denom

        # We'll do multiple passes to refine alpha & AR
        for _ in range(self.max_alpha_passes):
            self._update_ar_coeffs(data, gamma)
            if self.alpha_is_free:
                self._update_alpha(data, gamma)

        # (4) Covariances
        for j in range(self.n_regimes):
            sum_cov = np.zeros_like(self.covars_[j])
            count_j = 1e-300
            for t in range(n_samples):
                w = gamma[t, j]
                if t == 0:
                    pred_mean = self.means_[j]
                else:
                    diff_t_1 = data[t-1] - self.means_[j]
                    pred_mean = self.means_[j] + self.alpha*(self.ar_coeffs_[j] @ diff_t_1)
                resid = data[t] - pred_mean
                if self.covariance_type == 'full':
                    sum_cov += w * np.outer(resid, resid)
                else:  # diag
                    sum_cov += w * (resid**2)
                count_j += w
            sum_cov /= count_j
            self.covars_[j] = sum_cov

    def _update_ar_coeffs(self, data: np.ndarray, gamma: np.ndarray):
        """
        Weighted least squares for each state j:
          Y(t) = data[t]-means_[j],
          X(t) = (data[t-1]-means_[j]),
          Y(t) = alpha * A_j X(t).
        We'll solve A_j given alpha, for t=1..n_samples-1.
        """
        n_samples, d = data.shape
        for j in range(self.n_regimes):
            ZtZ = np.zeros((d, d))
            ZtY = np.zeros((d, d))
            for t in range(1, n_samples):
                w = gamma[t, j]
                if w < 1e-12:
                    continue
                xtm1 = data[t-1] - self.means_[j]
                y_t = data[t] - self.means_[j]
                z_t = self.alpha * xtm1  # shape (d,)
                ZtZ += w * np.outer(z_t, z_t)
                ZtY += w * np.outer(z_t, y_t)
            # pseudo-inverse
            if np.linalg.matrix_rank(ZtZ) < d:
                continue
            A_j = np.linalg.pinv(ZtZ) @ ZtY  # shape (d, d)
            self.ar_coeffs_[j] = A_j.T

    def _update_alpha(self, data: np.ndarray, gamma: np.ndarray):
        """
        Single global alpha across all states j:
          alpha = sum_{t,j} w[t,j]*(<y_t, A_j x_t>) / sum_{t,j} w[t,j]*(<A_j x_t, A_j x_t>)
        """
        n_samples, d = data.shape
        num = 0.0
        den = 1e-300
        for j in range(self.n_regimes):
            for t in range(1, n_samples):
                w = gamma[t, j]
                if w < 1e-12:
                    continue
                xtm1 = data[t-1] - self.means_[j]
                y_t = data[t] - self.means_[j]
                Ajx = self.ar_coeffs_[j] @ xtm1
                num += w * np.dot(y_t, Ajx)
                den += w * np.dot(Ajx, Ajx)
        self.alpha = num / den

    ########################################################################
    # Main Fit and Score
    ########################################################################
    def fit(self, data: pd.DataFrame):
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        self._init_params(data_array)

        old_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step
            log_likelihood = self._compute_log_likelihood(data_array)
            forward = self._forward_pass(log_likelihood)
            backward = self._backward_pass(log_likelihood)

            # compute gamma, xi

            ll = np.logaddexp.reduce(forward[-1])
            gamma, xi, _ = self._compute_posteriors(forward, backward, log_likelihood)

            # M-step
            self._update_parameters(data_array, gamma, xi)

            # Check convergence
            if abs(ll - old_ll) < self.tol:
                break
            old_ll = ll

        self.fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.Series:
        """
        Basic posterior-based decoding (not true Viterbi).
        """
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        log_likelihood = self._compute_log_likelihood(data_array)
        forward = self._forward_pass(log_likelihood)
        backward = self._backward_pass(log_likelihood)
        ll = np.logaddexp.reduce(forward[-1])

        # posteriors
        combined = forward + backward
        gamma = np.exp(combined - ll)
        states = np.argmax(gamma, axis=1)
        return pd.Series(states, index=data.index, name='Regime').apply(lambda x: f"regime{x+1}")

    def score(self, data: pd.DataFrame) -> float:
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        log_likelihood = self._compute_log_likelihood(data_array)
        forward = self._forward_pass(log_likelihood)
        return np.logaddexp.reduce(forward[-1])

    ########################################################################
    # Parameter Counting
    ########################################################################
    def num_params(self) -> int:
        """
        AR(1) HMM param count:
          - init: (n_regimes - 1)
          - transition: n_regimes*(n_regimes-1)
          - means: n_regimes*d
          - AR coeff: n_regimes*(d^2)
          - alpha (if free): +1
          - covars (full => n_regimes*(d(d+1)/2), diag => n_regimes*d)
        """
        if self.means_ is None:
            return 0
        n = self.n_regimes
        d = self.means_.shape[1]

        init_params = n - 1
        trans_params = n*(n-1)
        means_params = n*d
        ar_params = n*(d**2)
        alpha_params = 1 if self.alpha_is_free else 0

        if self.covariance_type == 'full':
            cov_params = n * ((d*(d+1)) // 2)
        else:
            cov_params = n*d

        return init_params + trans_params + means_params + ar_params + alpha_params + cov_params

    def num_params_empirical(self) -> int:
        """
        An alternate shape-based count, for debugging.
        """
        if self.means_ is None:
            return 0
        n = self.n_regimes
        init_params = n - 1
        trans_params = n*(n-1)

        d = self.means_.shape[1]
        means_params = self.means_.size
        ar_params = self.ar_coeffs_.size if self.ar_coeffs_ is not None else 0
        alpha_params = 1 if self.alpha_is_free else 0

        cov_params = 0
        if self.covars_ is not None:
            if self.covariance_type == 'full':
                for c in self.covars_:
                    d_ = c.shape[0]
                    cov_params += (d_*(d_+1))//2
            else:
                cov_params = self.covars_.size

        return init_params + trans_params + means_params + ar_params + cov_params + alpha_params






















def _log_gaussian(x, mean, cov):
    d = len(x)
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -1e15
    return -0.5 * (
        d*np.log(2*np.pi) +
        logdet +
        diff.T @ np.linalg.inv(cov) @ diff
    )

def _log_gaussian_diag(x, mean, diagvar):
    diff = x - mean
    var = diagvar
    log_prob = -0.5 * np.sum(np.log(2*np.pi*var) + (diff**2)/var)
    return log_prob
