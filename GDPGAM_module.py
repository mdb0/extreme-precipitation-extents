import numpy as np
import scipy as sp
from pygam import GAM
from pygam.distributions import Distribution
from pygam.utils import ylogydu
from functools import wraps

# --- 1. Helper Decorators (from pygam source) ---
def multiply_weights(deviance):
    @wraps(deviance)
    def multiplied(self, y, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return deviance(self, y, mu, **kwargs) * weights
    return multiplied

def divide_weights(V):
    @wraps(V)
    def divided(self, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return V(self, mu, **kwargs) / weights
    return divided

# --- 2. Custom GPD Distribution ---
class GPDDist(Distribution):
    """
    Generalized Pareto Distribution for pygam.
    Models the mean mu = sigma / (1 - xi).
    """
    def __init__(self, xi=0.1, scale=1.0):
        # We treat scale=1.0 fixed because GPD dispersion is controlled by xi
        super(GPDDist, self).__init__(name="gpd", scale=scale)
        self.xi = xi
        self._known_scale = True 

    def log_pdf(self, y, mu, weights=None):
        if weights is None:
            weights = np.ones_like(mu)
        
        # Convert Mean (mu) -> Scale (sigma)
        sigma = mu * (1 - self.xi)
        
        # GPD Log PDF
        # f(x) = (1/sigma) * (1 + xi*x/sigma)^(-1 - 1/xi)
        term = 1 + self.xi * (y / sigma)
        
        # Avoid log(negative)
        valid = (sigma > 0) & (term > 0)
        lp = np.full_like(y, -np.inf)
        
        lp[valid] = -np.log(sigma[valid]) - (1 + 1/self.xi) * np.log(term[valid])
        return lp * weights

    @divide_weights
    def V(self, mu):
        """
        Variance function. 
        For GPD, Var(Y) ~ mu^2. We use this relationship for the PIRLS weights.
        """
        return mu**2

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        GPD Deviance = 2 * (LogLikelihood_Saturated - LogLikelihood_Model)
        """
        # 1. Log-Likelihood of Model
        sigma = mu * (1 - self.xi)
        term_model = 1 + self.xi * (y / sigma)
        
        # Handle instability
        mask = (term_model > 0) & (sigma > 0) & (y > 0)
        ll_model = np.zeros_like(y)
        ll_model[mask] = -np.log(sigma[mask]) - (1 + 1/self.xi) * np.log(term_model[mask])
        ll_model[~mask] = -np.inf

        # 2. Log-Likelihood of Saturated Model (mu = y)
        # If mu = y, then sigma_sat = y * (1 - xi)
        # term_sat = 1 + xi * y / (y * (1 - xi)) = 1 + xi/(1-xi) = 1/(1-xi)
        # ll_sat = -log(y(1-xi)) - (1 + 1/xi) * log(1/(1-xi))
        #        = -log(y) - log(1-xi) + (1 + 1/xi) * log(1-xi)
        #        = -log(y) + (1/xi) * log(1-xi)
        
        ll_sat = np.zeros_like(y)
        ll_sat[mask] = -np.log(y[mask]) + (1/self.xi) * np.log(1 - self.xi)
        
        # Deviance
        dev = 2 * (ll_sat - ll_model)
        
        # Replace infinities/nans
        dev[~mask] = np.inf
        dev = np.nan_to_num(dev, nan=np.inf)
        
        if scaled:
            dev /= self.scale # scale is 1.0
            
        return dev

    def sample(self, mu):
        sigma = mu * (1 - self.xi)
        return sp.stats.genpareto.rvs(c=self.xi, scale=sigma)

# --- 3. Custom GPD GAM Class ---
class GPDGAM(GAM):
    """
    GAM that learns both the regression coefficients AND the shape parameter xi.
    It uses a Profile Likelihood approach:
    1. Fix xi, fit GAM (optimize beta).
    2. Fix beta, optimize xi (using residuals).
    3. Repeat until convergence.
    """
    def __init__(self, terms='auto', xi_init=0.1, **kwargs):
        # Initialize with our custom distribution
        dist = GPDDist(xi=xi_init)
        super().__init__(terms=terms, distribution=dist, link='log', **kwargs)
        
    def fit(self, X, y, weights=None, n_iter_profile=5, verbose=True):
        """
        Fit coefficients and learn xi iteratively.
        """
        y = np.asarray(y)
        X = np.asarray(X)
        
        if np.any(y <= 0):
             raise ValueError("Target y must be strictly positive excesses.")

        current_xi = self.distribution.xi
        
        print(f"Starting Profile Likelihood Optimization (Initial xi={current_xi})")
        
        for i in range(n_iter_profile):
            # 1. Fit the GAM coefficients (beta) given current xi
            # We call the parent fit method which runs PIRLS
            super().fit(X, y, weights=weights)
            
            # 2. Predict current mu
            mu_pred = self.predict(X, type='response')
            
            # 3. Estimate new xi from residuals
            # If Y ~ GPD(sigma=mu(1-xi), xi), then Y/mu ~ GPD(scale=1-xi, xi)
            # We treat Y/mu as i.i.d samples to update xi
            scaled_residuals = y / mu_pred
            
            # Use scipy to fit GPD to these residuals
            # We fix loc=0. We optimize shape (c) and scale.
            # We expect scale approx (1 - xi).
            params = sp.stats.genpareto.fit(scaled_residuals, floc=0)
            new_xi = params[0] # The shape parameter c is xi
            
            # Constrain xi to reasonable bounds for precipitation (e.g., < 0.5)
            new_xi = np.clip(new_xi, -0.5, 0.5)
            
            diff = abs(new_xi - current_xi)
            if verbose:
                print(f"Iter {i+1}: Updated xi from {current_xi:.4f} to {new_xi:.4f} (Diff: {diff:.4e})")
            
            # Update distribution state
            self.distribution.xi = new_xi
            current_xi = new_xi
            
            if diff < 1e-3:
                if verbose: print("Convergence reached.")
                break
                
        # Store final shape
        self.shape_xi_ = current_xi
        return self

    def predict(self, X, type='response'):
        # Override to handle 'scale' prediction specifically for GPD
        res = super().predict(X)
        
        if type == 'scale':
            # sigma = mu * (1 - xi)
            mu = super().predict(X)
            return mu * (1 - self.distribution.xi)
            
        return res

    def summary(self):
        super().summary()
        print("-" * 72)
        print(f"Final GPD Shape Parameter (xi): {self.distribution.xi:.4f}")
        print("-" * 72)
