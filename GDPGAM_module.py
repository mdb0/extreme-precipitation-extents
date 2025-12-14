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


## visu ##############################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gamma

def plot_return_levels(weather_df, temp_df, gam_gamma, gam_logit, gam_gpd):
    print("Generating Figure 6: Seasonal Return Levels...")

    # 1. Select Station (Try 110/478, else largest)
    target_stations = [110, 478]
    available_stations = weather_df['NUM_POSTE'].unique()
    
    station_id = None
    for tid in target_stations:
        if tid in available_stations:
            station_id = tid
            break
    
    if station_id is None:
        station_id = weather_df['NUM_POSTE'].value_counts().idxmax()
        print(f"Target stations {target_stations} not found. Using Station {station_id} instead.")
    else:
        print(f"Using Station {station_id} (Matches Paper).")

    # 2. Prepare Full Time Series for Prediction
    # Extract station constants
    station_data = weather_df[weather_df['NUM_POSTE'] == station_id].iloc[0]
    lon, lat, alti = station_data['LON'], station_data['LAT'], station_data['ALTI']

    # Create prediction dataframe from the FIXED temp_df
    pred_df = temp_df.copy()
    pred_df['day_of_year'] = pred_df['Date'].dt.dayofyear
    pred_df['year'] = pred_df['Date'].dt.year
    
    # Add Station Constants
    pred_df['LON'] = lon
    pred_df['LAT'] = lat
    pred_df['ALTI'] = alti
    
    # Feature Matrix X (Must match training columns!)
    X_pred = pred_df[['LON', 'LAT', 'ALTI', 'day_of_year', 'temp_30d_avg']].dropna().values
    
    # Filter pred_df to match X_pred length (dropping NaNs from rolling window)
    pred_df = pred_df.dropna(subset=['temp_30d_avg'])

    # 3. Predict Parameters
    print("Predicting daily parameters...")
    
    # A. Gamma Model (Threshold u)
    mu_gamma = gam_gamma.predict(X_pred)
    shape_gamma = 1 / gam_gamma.distribution.scale
    scale_gamma = mu_gamma / shape_gamma
    pred_df['u_hat'] = gamma.ppf(0.90, a=shape_gamma, scale=scale_gamma)
    
    # B. Logistic Model (Probability p)
    pred_df['p_hat'] = gam_logit.predict_proba(X_pred)
    
    # C. GPD Model (Scale sigma and Shape xi)
    if hasattr(gam_gpd, 'distribution') and hasattr(gam_gpd.distribution, 'xi'):
        xi_hat = gam_gpd.distribution.xi
    else:
        xi_hat = gam_gpd.shape_xi_
    
    pred_df['sigma_hat'] = gam_gpd.predict(X_pred, type='scale')
    
    # 4. Calculate Return Level (100-year daily)
    q = 1 / (100 * 365)
    term = (pred_df['p_hat'] / q) ** xi_hat
    pred_df['return_level'] = 10 + pred_df['u_hat'] + (pred_df['sigma_hat'] / xi_hat) * (term - 1)

    # 5. Seasonal Aggregation
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'
        
    pred_df['Season'] = pred_df['Date'].dt.month.apply(get_season)
    seasonal_rl = pred_df.groupby(['year', 'Season'])['return_level'].mean().reset_index()

    # 6. Plotting
    print("Plotting...")
    plt.figure(figsize=(8, 4))
    colors = {'Winter': 'purple', 'Spring': 'olive', 'Summer': 'cyan', 'Fall': 'pink'}
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        data = seasonal_rl[seasonal_rl['Season'] == season]
        plt.plot(data['year'], data['return_level'], label=season, color=colors[season], linewidth=2, alpha=0.8)
    
    plt.title(f"Estimated Seasonal Average Return Level (100-Year)\nStation {station_id}", fontsize=10)
    plt.ylabel("Return Level (mm)", fontsize=8)
    plt.xlabel("Year", fontsize=8)
    plt.legend(title="Season")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return seasonal_rl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_gamma_gam(gam_model):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Spatial Interaction (LON, LAT)', 'Elevation Effect', 'Seasonality (Day of Year)', 'Temperature Effect']

    # 1. Spatial Interaction (Tensor Product)
    # visualizing 2D interaction is tricky, we use a contour plot
    XX = gam_model.generate_X_grid(term=0, meshgrid=True)
    Z = gam_model.partial_dependence(term=0, X=XX, meshgrid=True)
    
    # We need the actual values for axes
    ax = axs[0]
    c = ax.contourf(XX[0], XX[1], Z, cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_title(titles[0])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 2. Elevation (Spline)
    XX = gam_model.generate_X_grid(term=1)
    pdep, confi = gam_model.partial_dependence(term=1, X=XX, width=0.95)
    
    ax = axs[1]
    ax.plot(XX[:, 2], pdep, label='Effect') # Index 2 is ALTI
    ax.plot(XX[:, 2], confi, c='r', ls='--', label='95% CI')
    ax.set_title(titles[1])
    ax.set_xlabel('Elevation (m)')
    ax.grid(True, alpha=0.3)

    # 3. Seasonality (Cyclic Spline)
    XX = gam_model.generate_X_grid(term=2)
    pdep, confi = gam_model.partial_dependence(term=2, X=XX, width=0.95)
    
    ax = axs[2]
    ax.plot(XX[:, 3], pdep) # Index 3 is day_of_year
    ax.plot(XX[:, 3], confi, c='r', ls='--')
    ax.set_title(titles[2])
    ax.set_xlabel('Day of Year')
    ax.grid(True, alpha=0.3)

    # 4. Temperature (Linear)
    XX = gam_model.generate_X_grid(term=3)
    pdep, confi = gam_model.partial_dependence(term=3, X=XX, width=0.95)
    
    ax = axs[3]
    ax.plot(XX[:, 4], pdep) # Index 4 is temp
    ax.plot(XX[:, 4], confi, c='r', ls='--')
    ax.set_title(titles[3])
    ax.set_xlabel('Basin Temp (Std)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_logistic_gam(gam_model):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Terms to plot: Elevation, Seasonality, Temperature
    # (Skipping spatial for brevity, but you can add it back like above)
    term_indices = [1, 2, 3] 
    labels = ['Elevation (m)', 'Day of Year', 'Temperature']
    feature_indices = [2, 3, 4] # Col index in X

    for i, term_idx in enumerate(term_indices):
        ax = axs[i]
        
        # Generate grid
        XX = gam_model.generate_X_grid(term=term_idx)
        
        # Get partial dependence (Log-Odds scale)
        pdep, confi = gam_model.partial_dependence(term=term_idx, X=XX, width=0.95)
        
        # Inverse Logit Transform to get Probability contribution centered at 0.5
        # Note: This is an approximation for visualization to show the shape
        prob_dep = 1 / (1 + np.exp(-pdep))
        prob_confi = 1 / (1 + np.exp(-confi))
        
        ax.plot(XX[:, feature_indices[i]], prob_dep, color='green', lw=2)
        ax.fill_between(XX[:, feature_indices[i]].flatten(), 
                        prob_confi[:, 0], prob_confi[:, 1], 
                        color='green', alpha=0.1)
        
        ax.set_title(f"Effect on Exceedance Probability\n({labels[i]})")
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Probability Contribution")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Logistic Model: Drivers of Extreme Events", fontsize=14)
    plt.tight_layout()
    plt.show()

import scipy.stats as stats
import inspect # Needed for the check inside the function

def check_gpd_fit(gam_model, y_true, X_input):
    """
    Diagnostic plots for GPD Model:
    1. Partial Dependence of Scale parameter (sigma)
    2. Q-Q Plot of Residuals (Model Fit Check)
    """
    
    # --- Part A: Partial Dependence (Scale Parameter Sigma) ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # 1. Seasonality Effect on Sigma
    XX = gam_model.generate_X_grid(term=2)
    pdep = gam_model.partial_dependence(term=2, X=XX) # Log scale
    sigma_effect = np.exp(pdep) # Convert log-link to multiplicative effect
    
    axs[0].plot(XX[:, 3], sigma_effect, color='purple')
    axs[0].set_title("Seasonality Effect on Scale ($\sigma$)")
    axs[0].set_ylabel("Multiplicative Factor")
    axs[0].set_xlabel("Day of Year")
    
    # 2. Temperature Effect on Sigma
    XX = gam_model.generate_X_grid(term=3)
    pdep = gam_model.partial_dependence(term=3, X=XX)
    sigma_effect = np.exp(pdep)
    
    axs[1].plot(XX[:, 4], sigma_effect, color='red')
    axs[1].set_title("Temperature Effect on Scale ($\sigma$)")
    axs[1].set_xlabel("Temperature (Std)")
    
    plt.show()

    # --- Part B: Q-Q Plot (The "Proof" it works) ---
    # We transform data to standard exponential using the fitted parameters
    # If Y ~ GPD(sigma, xi), then (1/xi) * log(1 + xi * Y/sigma) ~ Exponential(1)
    
    # Predict sigma for all points
    # Note: Use type='response' to get mean, then convert to sigma if using custom dist class
    # Or use our .predict(type='scale') method if using the previous GPDGAM class
    if hasattr(gam_model, 'predict') and 'scale' in inspect.signature(gam_model.predict).parameters:
         pred_sigma = gam_model.predict(X_input, type='scale')
    else:
         # Fallback for custom dist class
         pred_mu = gam_model.predict(X_input, type='response')
         pred_sigma = pred_mu * (1 - gam_model.distribution.xi)
            
    xi = gam_model.distribution.xi
    
    # Transform residuals
    # T(y) = (1/xi) * log(1 + xi * y / sigma)
    transformed_data = (1/xi) * np.log(1 + xi * (y_true / pred_sigma))
    
    # Q-Q Plot against standard Exponential
    plt.figure(figsize=(4, 4))
    stats.probplot(transformed_data, dist="expon", plot=plt)
    plt.title(f"Q-Q Plot of Transformed Residuals\n(Shape $\\xi$ = {xi:.3f})")
    plt.grid(True, alpha=0.3)
    plt.show()
