# hmm_models.py

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM,GMMHMM



class MultivariateGaussian:
    """
    Represents a multivariate Gaussian distribution.
    
    This class encapsulates the parameters of a multivariate Gaussian distribution
    and provides methods to compute probability density values.
    
    Parameters
    ----------
    means : numpy.ndarray
        Mean vector of the distribution, shape (d,) where d is the dimension
    cov : numpy.ndarray
        Covariance matrix, shape (d, d)
        
    Attributes
    ----------
    means : numpy.ndarray
        Mean vector
    cov : numpy.ndarray
        Covariance matrix
        
    Notes
    -----
    This class is used as a building block for more complex models like
    Gaussian HMMs and Gaussian Mixture Models.
    The probability density function is given by:
    p(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (x-μ)ᵀΣ^(-1)(x-μ))
    where μ is the mean vector, Σ is the covariance matrix, and d is the dimension.
    """


    def __init__(self, means, cov):
        self.means = means
        self.cov = cov

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at point x.
        
        Parameters
        ----------
        x : numpy.ndarray
            Point at which to evaluate the PDF, shape (d,)
            
        Returns
        -------
        float
            The probability density at point x
            
        Notes
        -----
        Uses scipy.stats.multivariate_normal for the calculation.
        """

        return multivariate_normal(mean=self.means, cov=self.cov).pdf(x)






class GaussianMixture:
    """
    Represents a Gaussian Mixture Model (GMM).
    
    A GMM is a weighted sum of multiple Gaussian distributions. This class
    stores the components and provides methods to compute probability density values.
    
    Parameters
    ----------
    weights : numpy.ndarray
        Mixture weights for each component, shape (n_components,)
    gaussians : list of MultivariateGaussian
        List of Gaussian components
        
    Attributes
    ----------
    weights : numpy.ndarray
        Mixture weights
    gaussians : list of MultivariateGaussian
        Gaussian components
        
    Notes
    -----
    This class is used to represent emission distributions in the GMMHMM model.
    The probability density function is given by:
    p(x) = Σ(w_i * p_i(x))
    where w_i are the mixture weights, and p_i(x) are the component densities.
    """

    def __init__(self, weights, gaussians):
        self.weights = weights
        self.gaussians = gaussians

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at point x.
        
        For a mixture model, the PDF is the weighted sum of component PDFs.
        
        Parameters
        ----------
        x : numpy.ndarray
            Point at which to evaluate the PDF, shape (d,)
            
        Returns
        -------
        float
            The probability density at point x
        """

        total = 0.0
        for w, g in zip(self.weights, self.gaussians):
            total += w * g.pdf(x)
        return total







class GaussianHMMWrapper:
    """
    Wrapper for hmmlearn's GaussianHMM with additional functionality.
    
    This class provides a consistent interface for working with Gaussian HMMs,
    with additional methods for state prediction, parameter counting, and more.
    
    A Gaussian HMM models time series data as transitions between hidden states,
    where each state emits observations according to a Gaussian distribution.
    
    Parameters
    ----------
    n_regimes : int
        Number of hidden states (regimes)
    covariance_type : str, default='full'
        Type of covariance parameter, one of 'full', 'diag'
    n_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-3
        Convergence threshold for EM algorithm
        
    Attributes
    ----------
    n_regimes : int
        Number of hidden states
    covariance_type : str
        Type of covariance matrices
    n_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    model : hmmlearn.hmm.GaussianHMM
        The underlying hmmlearn model
    regime_list : list of MultivariateGaussian
        List of emission distributions for each regime
    regime_dict : dict
        Mapping from regime names to distributions
    index_remap : dict
        Maps original indices to sorted regime indices
        
    Notes
    -----
    After fitting, regimes are sorted by their mean value on the first dimension
    and renamed to "regime1", "regime2", etc.
    
    The joint probability of observations X = (x₁,...,x_T) and hidden states Z = (z₁,...,z_T) is:
    p(X,Z) = p(z₁) * p(x₁|z₁) * Π[t=2 to T] p(z_t|z_{t-1}) * p(x_t|z_t)
    where p(z₁) is the initial state probability, p(z_t|z_{t-1}) are transition probabilities,
    and p(x_t|z_t) are emission probabilities modeled as Gaussians.
    """

    def __init__(self, n_regimes, covariance_type='full', n_iter=100, tol=1e-3):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
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
        """
        Fit the Gaussian HMM to time series data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to fit, with rows as time points and columns as variables
            
        Returns
        -------
        self
            The fitted model instance
            
        Notes
        -----
        After fitting, regimes are sorted by their mean on the first dimension and 
        indices are remapped accordingly.
        """

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
        """
        Predict the most likely hidden state sequence for the given data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data for prediction
            
        Returns
        -------
        pd.Series
            Series of predicted regime labels ("regime1", "regime2", etc.) with the
            same index as the input data
            
        Notes
        -----
        Uses the Viterbi algorithm to find the most likely state sequence.
        """

        hidden_states = self.model.predict(data)
        mapped = pd.Series(hidden_states, index=data.index).map(self.index_remap)
        return mapped.apply(lambda x: f"regime{x+1}")



    def score(self, data: pd.DataFrame) -> float:
        """
        Compute the log-likelihood of the data under this model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to score
            
        Returns
        -------
        float
            Log-likelihood score
        """

        return self.model.score(data)

    def num_params(self) -> int:
        """
        Calculate the number of free parameters in the model.
        
        For a Gaussian HMM, the parameters include:
        - Initial state probabilities: n_regimes - 1
        - Transition probabilities: n_regimes * (n_regimes - 1)
        - Emission means: n_regimes * n_features
        - Emission covariances: depends on covariance_type
          - 'full': n_regimes * (n_features * (n_features + 1)) / 2
          - 'diag': n_regimes * n_features
            
        Returns
        -------
        int
            Total number of free parameters
            
        Notes
        -----
        This count is used for computing information criteria like AIC and BIC.
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
        Calculate parameter count by directly examining array shapes.
        
        This alternative counting method inspects the actual arrays in the
        fitted model. It serves as a sanity check for the theoretical count.
        
        Returns
        -------
        int
            Total number of free parameters based on array dimensions
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
    Wrapper for hmmlearn's GMMHMM with additional functionality.
    
    This class provides a consistent interface for working with Gaussian Mixture HMMs,
    with additional methods for state prediction, parameter counting, and more.
    
    A GMM-HMM models time series data as transitions between hidden states,
    where each state emits observations according to a Gaussian Mixture Model.
    
    Parameters
    ----------
    n_regimes : int
        Number of hidden states (regimes)
    n_mix : int, default=2
        Number of mixture components per state
    covariance_type : str, default='full'
        Type of covariance parameter, one of 'full', 'diag'
    n_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-3
        Convergence threshold for EM algorithm
        
    Attributes
    ----------
    n_regimes : int
        Number of hidden states
    n_mix : int
        Number of mixture components per state
    covariance_type : str
        Type of covariance matrices
    n_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    model : hmmlearn.hmm.GMMHMM
        The underlying hmmlearn model
    index_remap : dict
        Maps original indices to sorted regime indices
    regime_list : list of GaussianMixture
        List of emission distributions for each regime
    regime_dict : dict
        Mapping from regime names to distributions
        
    Notes
    -----
    After fitting, regimes are sorted by their mean value on the first dimension
    and renamed to "regime1", "regime2", etc.
    
    The emission probability for state j is given by a Gaussian mixture:
    p(x_t|z_t=j) = Σ(w_{jk} * N(x_t|μ_{jk},Σ_{jk}))
    where w_{jk}, μ_{jk}, and Σ_{jk} are the weight, mean, and covariance 
    of the k-th component in state j.
    """
    
    def __init__(self, n_regimes, n_mix=2, covariance_type='full', n_iter=100, tol=1e-3):
        self.n_regimes = n_regimes
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
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
        """
        Fit the GMM-HMM to time series data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to fit, with rows as time points and columns as variables
            
        Returns
        -------
        self
            The fitted model instance
            
        Notes
        -----
        After fitting, regimes are sorted by their mean on the first dimension and 
        indices are remapped accordingly.
        """

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
        """
        Predict the most likely hidden state sequence for the given data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data for prediction
            
        Returns
        -------
        pd.Series
            Series of predicted regime labels ("regime1", "regime2", etc.) with the
            same index as the input data
            
        Notes
        -----
        Uses the Viterbi algorithm to find the most likely state sequence.
        """

        hidden_states = self.model.predict(data)
        mapped = pd.Series(hidden_states, index=data.index).map(self.index_remap)
        return mapped.apply(lambda x: f"regime{x+1}")

    def score(self, data: pd.DataFrame) -> float:
        """
        Compute the log-likelihood of the data under this model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to score
            
        Returns
        -------
        float
            Log-likelihood score
        """

        return self.model.score(data)

    def num_params(self) -> int:
        """
        Calculate the number of free parameters in the model.
        
        For a GMM-HMM, the parameters include:
        - Initial state probabilities: n_regimes - 1
        - Transition probabilities: n_regimes * (n_regimes - 1)
        - Mixture weights: n_regimes * (n_mix - 1)
        - Emission means: n_regimes * n_mix * n_features
        - Emission covariances: depends on covariance_type
          - 'full': n_regimes * n_mix * (n_features * (n_features + 1)) / 2
          - 'diag': n_regimes * n_mix * n_features
            
        Returns
        -------
        int
            Total number of free parameters
            
        Notes
        -----
        This count is used for computing information criteria like AIC and BIC.
        """

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
        """
        Calculate parameter count by directly examining array shapes.
        
        This alternative counting method inspects the actual arrays in the
        fitted model. It serves as a sanity check for the theoretical count.
        
        Returns
        -------
        int
            Total number of free parameters based on array dimensions
        """

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
    Autoregressive Hidden Markov Model with Gaussian emissions.
    
    This class implements an AR(1) HMM with state-dependent autoregressive coefficients.
    The model follows the formula:
        X_t = means[j] + alpha * (ar_coeffs_[j] @ X_{t-1}) + noise
    
    where j is the hidden state at time t, means[j] is the state-specific mean,
    ar_coeffs_[j] is the state-specific AR coefficient matrix, and alpha is a
    global AR coefficient that can be either fixed or learned.
    
    Parameters
    ----------
    n_regimes : int
        Number of hidden states (regimes)
    covariance_type : str, default='full'
        Type of covariance matrices, one of 'full', 'diag'
    n_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-3
        Convergence threshold for EM algorithm
    alpha : float, default=1.0
        Initial value for the global AR coefficient
    alpha_is_free : bool, default=False
        Whether to learn alpha or keep it fixed
    p : int, default=1
        AR order (currently only p=1 is implemented)
    floor_prob : float, default=1e-12
        Minimal probability floor to avoid log(0) issues
    max_alpha_passes : int, default=2
        Number of times to refine alpha and AR in each M-step
        
    Attributes
    ----------
    n_regimes : int
        Number of hidden states
    covariance_type : str
        Type of covariance matrices
    n_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    alpha : float
        Global AR coefficient
    alpha_is_free : bool
        Whether alpha is learned
    p : int
        AR order
    floor_prob : float
        Probability floor
    max_alpha_passes : int
        Alpha refinement passes
    means_ : numpy.ndarray
        State-specific means, shape (n_regimes, n_features)
    covars_ : numpy.ndarray
        State-specific covariance matrices
    ar_coeffs_ : numpy.ndarray
        State-specific AR coefficient matrices, shape (n_regimes, n_features, n_features)
    transmat_ : numpy.ndarray
        Transition probability matrix, shape (n_regimes, n_regimes)
    startprob_ : numpy.ndarray
        Initial state probabilities, shape (n_regimes,)
    fitted_ : bool
        Whether the model has been fitted
        
    Notes
    -----
    This class implements the EM algorithm from scratch instead of using hmmlearn.
    It performs forward-backward passes to compute exact posteriors, and uses
    specialized M-step updates for the autoregressive coefficients.
    
    
    Forward-Backward Algorithm:
    - Forward recursion:
        α_t(j) = p(x_1,...,x_t, z_t=j)
        α_1(j) = p(z_1=j) * p(x_1|z_1=j)
        α_t(j) = p(x_t|z_t=j) * Σ_i [α_{t-1}(i) * p(z_t=j|z_{t-1}=i)]
    - Backward recursion:
        β_t(j) = p(x_{t+1},...,x_T | z_t=j)
        β_T(j) = 1
        β_t(j) = Σ_i [p(z_{t+1}=i|z_t=j) * p(x_{t+1}|z_{t+1}=i) * β_{t+1}(i)]
        
    For posterior computation:
    - State posterior:
        γ_t(j) = p(z_t=j|x_1,...,x_T) = α_t(j) * β_t(j) / p(x_1,...,x_T)
    - Transition posterior:
        ξ_t(i,j) = p(z_t=i, z_{t+1}=j|x_1,...,x_T) 
                = α_t(i) * p(z_{t+1}=j|z_t=i) * p(x_{t+1}|z_{t+1}=j) * β_{t+1}(j) / p(x_1,...,x_T)
    
    For the AR coefficient and alpha updates:
    - The AR coefficients A_j are updated using weighted least squares:
        A_j = (Σ_t γ_t(j) * x_{t-1} * x_{t-1}ᵀ)^(-1) * (Σ_t γ_t(j) * x_{t-1} * (x_t - μ_j)ᵀ)

    - The global AR coefficient α is updated as:
        α = Σ_{j,t} γ_t(j) * (x_t - μ_j)ᵀ * (A_j * x_{t-1}) / Σ_{j,t} γ_t(j) * (A_j * x_{t-1})ᵀ * (A_j * x_{t-1})
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
        """
        Initialize model parameters before EM.
        
        Sets up initial values for means, AR coefficients, covariances,
        transition matrix, and initial state probabilities.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data, shape (n_samples, n_features)
            
        Returns
        -------
        None
            Initializes various model attributes
        """

        n_features = data.shape[1]

        # means: shape (n_regimes, n_features)
        self.means_ = np.random.randn(self.n_regimes, n_features)
        # AR coeffs: shape (n_regimes, n_features, n_features)
        # Each A_j is (d x d), so (ar_coeffs_[j] @ X_{t-1}) has shape (d,)
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
        Compute log-likelihood matrix for the E-step.
        
        Calculates log p(x_t | state=j) for each time point t and state j.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data, shape (n_samples, n_features)
            
        Returns
        -------
        numpy.ndarray
            Log-likelihood matrix, shape (n_samples, n_regimes)
            
        Notes
        -----
        For t=0, we use a standard Gaussian likelihood.
        For t>0, we use an AR(1) prediction with state-specific coefficients.
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
                # AR(1) prediction: mu_j + alpha * (A_j @ X_{t-1})
                ar_mean = self.means_[j] + self.alpha * (self.ar_coeffs_[j] @ data[t - 1])
                if self.covariance_type == 'full':
                    log_likelihood[t, j] = _log_gaussian(data[t], ar_mean, self.covars_[j])
                else:
                    log_likelihood[t, j] = _log_gaussian_diag(data[t], ar_mean, self.covars_[j])

        return log_likelihood

    def _forward_pass(self, log_likelihood: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the forward-backward algorithm.
        
        Computes forward probabilities alpha(t,j) = p(x_0...x_t, state_t=j)
        
        Parameters
        ----------
        log_likelihood : numpy.ndarray
            Log-likelihood matrix from _compute_log_likelihood, shape (n_samples, n_regimes)
            
        Returns
        -------
        numpy.ndarray
            Forward log-probabilities, shape (n_samples, n_regimes)
        """

        n_samples, n_states = log_likelihood.shape
        forward = np.full((n_samples, n_states), -np.inf)

        # init
        for j in range(n_states):
            forward[0, j] = np.log(self.startprob_[j]) + log_likelihood[0, j]

        # recursion
        for t in range(1, n_samples):
            for j in range(n_states):
                temp = forward[t - 1] + np.log(self.transmat_[:, j])
                forward[t, j] = np.logaddexp.reduce(temp) + log_likelihood[t, j]

        return forward

    def _backward_pass(self, log_likelihood: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the forward-backward algorithm.
        
        Computes backward probabilities beta(t,j) = p(x_{t+1}...x_T | state_t=j)
        
        Parameters
        ----------
        log_likelihood : numpy.ndarray
            Log-likelihood matrix from _compute_log_likelihood, shape (n_samples, n_regimes)
            
        Returns
        -------
        numpy.ndarray
            Backward log-probabilities, shape (n_samples, n_regimes)
        """
        n_samples, n_states = log_likelihood.shape
        backward = np.full((n_samples, n_states), -np.inf)

        # init
        backward[-1] = 0.0  # log(1)

        # recursion
        for t in range(n_samples - 2, -1, -1):
            for i in range(n_states):
                temp = np.log(self.transmat_[i]) + log_likelihood[t + 1] + backward[t + 1]
                backward[t, i] = np.logaddexp.reduce(temp)

        return backward

    def _compute_posteriors(self, forward: np.ndarray, backward: np.ndarray, log_likelihood: np.ndarray):
        """
        Compute posterior probabilities for the E-step.
        
        Parameters
        ----------
        forward : numpy.ndarray
            Forward log-probabilities from _forward_pass
        backward : numpy.ndarray
            Backward log-probabilities from _backward_pass
        log_likelihood : numpy.ndarray
            Log-likelihood matrix from _compute_log_likelihood
            
        Returns
        -------
        tuple
            gamma, xi, ll:
            - gamma: State posteriors p(state_t=j | all data), shape (n_samples, n_regimes)
            - xi: Transition posteriors p(state_t=i, state_{t+1}=j | all data), 
              shape (n_samples-1, n_regimes, n_regimes)
            - ll: Total log-likelihood of the data
            
        Notes
        -----
        gamma[t,j] is the probability of being in state j at time t given all data.
        xi[t,i,j] is the probability of transitioning from state i at time t to 
        state j at time t+1, given all data.
        """
        n_samples, n_states = forward.shape
        ll = np.logaddexp.reduce(forward[-1])  # log p(data)
        gamma = np.exp(forward + backward - ll)

        # xi: shape (n_samples-1, n_states, n_states)
        xi = np.zeros((n_samples - 1, n_states, n_states))
        for t in range(n_samples - 1):
            temp = (
                forward[t, :, None]
                + np.log(self.transmat_)
                + backward[t + 1, None, :]
                + np.expand_dims(log_likelihood[t + 1], axis=0)
            )
            max_val = np.max(temp)
            xi_t = np.exp(temp - max_val)
            xi_sum = np.sum(xi_t)
            xi[t] = xi_t / (xi_sum + 1e-300)

        return gamma, xi, ll

    ########################################################################
    # M-step
    ########################################################################
    def _update_parameters(self, data: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """
        Update all model parameters in the M-step.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data, shape (n_samples, n_features)
        gamma : numpy.ndarray
            State posteriors from _compute_posteriors
        xi : numpy.ndarray
            Transition posteriors from _compute_posteriors
            
        Returns
        -------
        None
            Updates model parameters in-place
            
        Notes
        -----
        This updates:
        1. Initial state probabilities
        2. Transition matrix
        3. State-specific means
        4. AR coefficients and alpha
        5. State-specific covariances
        """

        n_samples, d = data.shape

        # (1) Start probs
        start = gamma[0] + self.floor_prob
        start /= np.sum(start)
        self.startprob_ = start

        # (2) Transition matrix
        trans_counts = np.sum(xi, axis=0)
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

        # (4) Repeated passes to refine alpha & A_j
        for _ in range(self.max_alpha_passes):
            self._update_ar_coeffs(data, gamma)
            if self.alpha_is_free:
                self._update_alpha(data, gamma)

        # (5) Covariances
        for j in range(self.n_regimes):
            sum_cov = np.zeros_like(self.covars_[j])
            count_j = 1e-300
            for t in range(n_samples):
                w = gamma[t, j]
                if t == 0:
                    # At t=0, we assume X(0) ~ N(means_[j], covars_[j])
                    pred_mean = self.means_[j]
                else:
                    pred_mean = self.means_[j] + self.alpha * (self.ar_coeffs_[j] @ data[t - 1])
                resid = data[t] - pred_mean
                if self.covariance_type == 'full':
                    sum_cov += w * np.outer(resid, resid)
                else:
                    sum_cov += w * (resid ** 2)
                count_j += w
            sum_cov /= count_j
            self.covars_[j] = sum_cov

    def _update_ar_coeffs(self, data: np.ndarray, gamma: np.ndarray):
        """
        Update the autoregressive coefficients in the M-step.
        
        Uses weighted least squares to estimate state-specific AR coefficients.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data, shape (n_samples, n_features)
        gamma : numpy.ndarray
            State posteriors
            
        Returns
        -------
        None
            Updates ar_coeffs_ in-place
            
        Notes
        -----
        For each state j, solves for A_j in the equation:
            (X_t - mu_j) = alpha * A_j X_{t-1}
        using weighted least squares with gamma[t,j] as weights.
        """

        n_samples, d = data.shape
        for j in range(self.n_regimes):
            ZtZ = np.zeros((d, d))
            ZtY = np.zeros((d, d))

            for t in range(1, n_samples):
                w = gamma[t, j]
                if w < 1e-12:
                    continue

                # Y(t) = X(t) - mu_j
                y_t = data[t] - self.means_[j]   # shape (d,)

                # Z(t) = alpha * X(t-1)
                x_tminus1 = data[t - 1]          # shape (d,)
                z_t = self.alpha * x_tminus1     # shape (d,)

                # Weighted outer products
                ZtZ += w * np.outer(z_t, z_t)  # shape (d,d)
                ZtY += w * np.outer(z_t, y_t)

            if np.linalg.matrix_rank(ZtZ) < d:
                # If rank-deficient, skip updating this state's A_j
                continue

            # Solve for A_j in shape (d, d)
            # Y(t) ~ A_j Z(t)
            A_j = np.linalg.pinv(ZtZ) @ ZtY  # shape (d,d)
            self.ar_coeffs_[j] = A_j  # no .T needed if we want A_j @ x_{t-1}

    def _update_alpha(self, data: np.ndarray, gamma: np.ndarray):
        """
        Update the global AR coefficient alpha in the M-step.
        
        Only used when alpha_is_free=True.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data, shape (n_samples, n_features)
        gamma : numpy.ndarray
            State posteriors
            
        Returns
        -------
        None
            Updates alpha in-place
            
        Notes
        -----
        Uses a weighted approach across all states to compute a single global alpha.
        """

        n_samples, d = data.shape
        num = 0.0
        den = 1e-300

        for j in range(self.n_regimes):
            for t in range(1, n_samples):
                w = gamma[t, j]
                if w < 1e-12:
                    continue

                y_t = data[t] - self.means_[j]   # shape (d,)
                x_tminus1 = data[t - 1]
                Ajx = self.ar_coeffs_[j] @ x_tminus1  # shape (d,)

                num += w * np.dot(y_t, Ajx)
                den += w * np.dot(Ajx, Ajx)

        self.alpha = num / den


    ########################################################################
    # Main Fit and Score
    ########################################################################

    def fit(self, data: pd.DataFrame):
        """
        Fit the AR-HMM to time series data using EM.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data, with rows as time points and columns as variables
            
        Returns
        -------
        self
            The fitted model instance
            
        Notes
        -----
        The fitting procedure:
        1. Initializes parameters
        2. Runs EM iterations until convergence or max_iter
        3. For each iteration:
           a. E-step: Compute log-likelihoods, forward-backward, posteriors
           b. M-step: Update all model parameters
        """

        data_array = data.values if isinstance(data, pd.DataFrame) else data
        self._init_params(data_array)

        old_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step
            log_likelihood = self._compute_log_likelihood(data_array)
            forward = self._forward_pass(log_likelihood)
            backward = self._backward_pass(log_likelihood)

            gamma, xi, ll = self._compute_posteriors(forward, backward, log_likelihood)

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
        Predict the most likely state sequence for the given data.
        
        Uses posterior decoding (not Viterbi algorithm) to assign states.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data for prediction
            
        Returns
        -------
        pd.Series
            Series of predicted regime labels ("regime1", "regime2", etc.) with the
            same index as the input data
        """

        data_array = data.values if isinstance(data, pd.DataFrame) else data
        log_likelihood = self._compute_log_likelihood(data_array)
        forward = self._forward_pass(log_likelihood)
        backward = self._backward_pass(log_likelihood)
        ll = np.logaddexp.reduce(forward[-1])

        gamma = np.exp(forward + backward - ll)
        states = np.argmax(gamma, axis=1)
        return pd.Series(states, index=data.index, name='Regime').apply(lambda x: f"regime{x+1}")

    def score(self, data: pd.DataFrame) -> float:
        """
        Compute the log-likelihood of the data under this model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to score
            
        Returns
        -------
        float
            Log-likelihood score
        """

        data_array = data.values if isinstance(data, pd.DataFrame) else data
        log_likelihood = self._compute_log_likelihood(data_array)
        forward = self._forward_pass(log_likelihood)
        return np.logaddexp.reduce(forward[-1])


    ########################################################################
    # Parameter Counting
    ########################################################################
    
    def num_params(self) -> int:
        """
        Calculate the number of free parameters in the model.
        
        For an AR-HMM, the parameters include:
        - Initial state probabilities: n_regimes - 1
        - Transition probabilities: n_regimes * (n_regimes - 1)
        - State means: n_regimes * n_features
        - AR coefficients: n_regimes * (n_features * n_features)
        - Alpha (if free): 1 or 0
        - Covariances: depends on covariance_type
          - 'full': n_regimes * (n_features * (n_features + 1)) / 2
          - 'diag': n_regimes * n_features
            
        Returns
        -------
        int
            Total number of free parameters
            
        Notes
        -----
        This count is used for computing information criteria like AIC and BIC.
        """

        if self.means_ is None:
            return 0
        n = self.n_regimes
        d = self.means_.shape[1]

        init_params = n - 1
        trans_params = n * (n - 1)
        means_params = n * d
        ar_params = n * (d ** 2)
        alpha_params = 1 if self.alpha_is_free else 0

        if self.covariance_type == 'full':
            cov_params = n * (d * (d + 1) // 2)
        else:
            cov_params = n * d

        return init_params + trans_params + means_params + ar_params + alpha_params + cov_params

    def num_params_empirical(self) -> int:
        """
        Calculate parameter count by directly examining array shapes.
        
        This alternative counting method inspects the actual arrays in the
        fitted model. It serves as a sanity check for the theoretical count.
        
        Returns
        -------
        int
            Total number of free parameters based on array dimensions
        """
        if self.means_ is None:
            return 0
        n = self.n_regimes
        init_params = n - 1
        trans_params = n * (n - 1)

        d = self.means_.shape[1]
        means_params = self.means_.size
        ar_params = self.ar_coeffs_.size if self.ar_coeffs_ is not None else 0
        alpha_params = 1 if self.alpha_is_free else 0

        cov_params = 0
        if self.covars_ is not None:
            if self.covariance_type == 'full':
                for c in self.covars_:
                    d_ = c.shape[0]
                    cov_params += (d_ * (d_ + 1)) // 2
            else:
                cov_params = self.covars_.size

        return init_params + trans_params + means_params + ar_params + cov_params + alpha_params













def _log_gaussian(x, mean, cov):
    """
    Compute log probability density for a multivariate Gaussian.
    
    Parameters
    ----------
    x : numpy.ndarray
        Point at which to evaluate, shape (d,)
    mean : numpy.ndarray
        Mean vector, shape (d,)
    cov : numpy.ndarray
        Covariance matrix, shape (d, d)
        
    Returns
    -------
    float
        Log probability density at point x
        
    Notes
    -----
    Implements the formula:
        log p(x) = -0.5 * (d*log(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ))
    where d is the dimension, μ is the mean, and Σ is the covariance.
    
    A small value (-1e15) is returned if the covariance determinant is non-positive.
    """

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
    """
    Compute log probability density for a Gaussian with diagonal covariance.
    
    Parameters
    ----------
    x : numpy.ndarray
        Point at which to evaluate, shape (d,)
    mean : numpy.ndarray
        Mean vector, shape (d,)
    diagvar : numpy.ndarray
        Diagonal of covariance matrix, shape (d,)
        
    Returns
    -------
    float
        Log probability density at point x
        
    Notes
    -----
    Implements the formula for diagonal covariance:
        log p(x) = -0.5 * sum(log(2πσ²) + (x-μ)²/σ²)
    where σ² is the variance for each dimension.
    
    This is more efficient than the full covariance computation when the
    covariance matrix is diagonal.
    """

    diff = x - mean
    var = diagvar
    log_prob = -0.5 * np.sum(np.log(2*np.pi*var) + (diff**2)/var)
    return log_prob
