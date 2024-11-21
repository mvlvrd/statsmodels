"""
Markov switching regression models

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
from scipy.special import logsumexp

import statsmodels.base.wrapper as wrap

from statsmodels.tsa.regime_switching import markov_switching


class MarkovRegression(markov_switching.MarkovSwitching):
    r"""
    First-order k-regime Markov switching regression model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : int
        The number of regimes.
    trend : {'n', 'c', 't', 'ct'}
        Whether or not to include a trend. To include an intercept, time trend,
        or both, set `trend='c'`, `trend='t'`, or `trend='ct'`. For no trend,
        set `trend='n'`. Default is an intercept.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : int, optional
        The order of the model describes the dependence of the likelihood on
        previous regimes. This depends on the model in question and should be
        set appropriately by subclasses.
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.
    switching_trend : bool or iterable, optional
        If a boolean, sets whether or not all trend coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of trend variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_exog : bool or iterable, optional
        If a boolean, sets whether or not all regression coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of exogenous variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_variance : bool, optional
        Whether or not there is regime-specific heteroskedasticity, i.e.
        whether or not the error term has a switching variance. Default is
        False.

    Notes
    -----
    This model is new and API stability is not guaranteed, although changes
    will be made in a backwards compatible way if possible.

    The model can be written as:

    .. math::

        y_t = a_{S_t} + x_t' \beta_{S_t} + \varepsilon_t \\
        \varepsilon_t \sim N(0, \sigma_{S_t}^2)

    i.e. the model is a dynamic linear regression where the coefficients and
    the variance of the error term may be switching across regimes.

    The `trend` is accommodated by prepending columns to the `exog` array. Thus
    if `trend='c'`, the passed `exog` array should not already have a column of
    ones.

    See the notebook `Markov switching dynamic regression
    <../examples/notebooks/generated/markov_regression.html>`__ for an
    overview.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    def __init__(self, endog, k_regimes, trend='c', exog=None, order=0,
                 exog_tvtp=None, switching_trend=True, switching_exog=True,
                 switching_variance=False, dates=None, freq=None,
                 missing='none'):

        # Properties
        from statsmodels.tools.validation import string_like
        self.trend = string_like(trend, "trend", options=("n", "c", "ct", "t"))
        self.switching_trend = switching_trend
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance

        # Exogenous data
        self.k_exog, exog = markov_switching.prepare_exog(exog)

        # Trend
        nobs = len(endog)
        self.k_trend = 0
        self._k_exog = self.k_exog
        trend_exog = None
        if trend == 'c':
            trend_exog = np.ones((nobs, 1))
            self.k_trend = 1
        elif trend == 't':
            trend_exog = (np.arange(nobs) + 1)[:, np.newaxis]
            self.k_trend = 1
        elif trend == 'ct':
            trend_exog = np.c_[np.ones((nobs, 1)),
                               (np.arange(nobs) + 1)[:, np.newaxis]]
            self.k_trend = 2
        if trend_exog is not None:
            exog = trend_exog if exog is None else np.c_[trend_exog, exog]
            self._k_exog += self.k_trend

        # Initialize the base model
        super().__init__(
            endog, k_regimes, order=order, exog_tvtp=exog_tvtp, exog=exog,
            dates=dates, freq=freq, missing=missing)

        # Switching options
        if self.switching_trend is True or self.switching_trend is False:
            self.switching_trend = [self.switching_trend] * self.k_trend
        elif not len(self.switching_trend) == self.k_trend:
            raise ValueError('Invalid iterable passed to `switching_trend`.')
        if self.switching_exog is True or self.switching_exog is False:
            self.switching_exog = [self.switching_exog] * self.k_exog
        elif not len(self.switching_exog) == self.k_exog:
            raise ValueError('Invalid iterable passed to `switching_exog`.')

        self.switching_coeffs = (
            np.r_[self.switching_trend,
                  self.switching_exog].astype(bool).tolist())

        # Parameters
        self.parameters['exog'] = self.switching_coeffs
        self.parameters['variance'] = [1] if self.switching_variance else [0]

    def predict_conditional(self, params):
        """
        In-sample prediction, conditional on the current regime

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform prediction.

        Returns
        -------
        predict : array_like
            Array of predictions conditional on current, and possibly past,
            regimes
        """
        params = np.array(params, ndmin=1)

        # Since in the base model the values are the same across columns, we
        # only compute a single column, and then expand it below.
        predict = np.zeros((self.k_regimes, self.nobs), dtype=params.dtype)

        for i in range(self.k_regimes):
            # Predict
            if self._k_exog > 0:
                coeffs = params[self.parameters[i, 'exog']]
                predict[i] = np.dot(self.exog, coeffs)

        return predict[:, None, :]

    def _resid(self, params):
        predict = np.repeat(self.predict_conditional(params),
                            self.k_regimes, axis=1)
        return self.endog - predict

    def _conditional_loglikelihoods(self, params):
        """
        Compute loglikelihoods conditional on the current period's regime
        """

        # Get residuals
        resid = self._resid(params)

        # Compute the conditional likelihoods
        variance = params[self.parameters['variance']].squeeze()
        if self.switching_variance:
            variance = np.reshape(variance, (self.k_regimes, 1, 1))

        conditional_loglikelihoods = (
            -0.5 * resid**2 / variance - 0.5 * np.log(2 * np.pi * variance))

        return conditional_loglikelihoods

    @property
    def _res_classes(self):
        return {'fit': (MarkovRegressionResults,
                        MarkovRegressionResultsWrapper)}

    def _em_iteration(self, params0):
        """
        EM iteration

        Notes
        -----
        This uses the inherited _em_iteration method for computing the
        non-TVTP transition probabilities and then performs the EM step for
        regression coefficients and variances.
        """
        # Inherited parameters
        result, params1 = super()._em_iteration(params0)

        tmp = np.sqrt(result.smoothed_marginal_probabilities)

        # Regression coefficients
        coeffs = None
        if self._k_exog > 0:
            coeffs = self._em_exog(result, self.endog, self.exog,
                                   self.parameters.switching['exog'], tmp)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'exog']] = coeffs[i]

        # Variances
        params1[self.parameters['variance']] = self._em_variance(
            result, self.endog, self.exog, coeffs, tmp)
        # params1[self.parameters['variance']] = 0.33282116

        return result, params1

    def _em_exog(self, result, endog, exog, switching, tmp=None):
        """
        EM step for regression coefficients
        """
        k_exog = exog.shape[1]
        coeffs = np.zeros((self.k_regimes, k_exog))

        # First, estimate non-switching coefficients
        if not np.all(switching):
            nonswitching_exog = exog[:, ~switching]
            nonswitching_coeffs = (
                np.dot(np.linalg.pinv(nonswitching_exog), endog))
            coeffs[:, ~switching] = nonswitching_coeffs
            endog = endog - np.dot(nonswitching_exog, nonswitching_coeffs)

        # Next, get switching coefficients
        if np.any(switching):
            switching_exog = exog[:, switching]
            if tmp is None:
                tmp = np.sqrt(result.smoothed_marginal_probabilities)
            for i in range(self.k_regimes):
                tmp_endog = tmp[i] * endog
                tmp_exog = tmp[i][:, np.newaxis] * switching_exog
                coeffs[i, switching] = (
                    np.dot(np.linalg.pinv(tmp_exog), tmp_endog))

        return coeffs

    def _em_variance(self, result, endog, exog, betas, tmp=None):
        """
        EM step for variances
        """
        k_exog = 0 if exog is None else exog.shape[1]

        if self.switching_variance:
            variance = np.zeros(self.k_regimes)
            for i in range(self.k_regimes):
                if k_exog > 0:
                    resid = endog - np.dot(exog, betas[i])
                else:
                    resid = endog
                variance[i] = (
                    np.sum(resid**2 *
                           result.smoothed_marginal_probabilities[i]) /
                    np.sum(result.smoothed_marginal_probabilities[i]))
        else:
            variance = 0
            if tmp is None:
                tmp = np.sqrt(result.smoothed_marginal_probabilities)
            for i in range(self.k_regimes):
                tmp_endog = tmp[i] * endog
                if k_exog > 0:
                    tmp_exog = tmp[i][:, np.newaxis] * exog
                    resid = tmp_endog - np.dot(tmp_exog, betas[i])
                else:
                    resid = tmp_endog
                variance += np.sum(resid**2)
            variance /= self.nobs
        return variance

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.

        Notes
        -----
        These are not very sophisticated and / or good. We set equal transition
        probabilities and interpolate regression coefficients between zero and
        the OLS estimates, where the interpolation is based on the regime
        number. We rely heavily on the EM algorithm to quickly find much better
        starting parameters, which are then used by the typical scoring
        approach.
        """
        # Inherited parameters
        params = markov_switching.MarkovSwitching.start_params.fget(self)

        # Regression coefficients
        if self._k_exog > 0:
            beta = np.dot(np.linalg.pinv(self.exog), self.endog)
            variance = np.var(self.endog - np.dot(self.exog, beta))

            if np.any(self.switching_coeffs):
                for i in range(self.k_regimes):
                    params[self.parameters[i, 'exog']] = (
                        beta * (i / self.k_regimes))
            else:
                params[self.parameters['exog']] = beta
        else:
            variance = np.var(self.endog)

        # Variances
        if self.switching_variance:
            params[self.parameters['variance']] = (
                np.linspace(variance / 10., variance, num=self.k_regimes))
        else:
            params[self.parameters['variance']] = variance

        return params

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        # Inherited parameters
        param_names = np.array(
            markov_switching.MarkovSwitching.param_names.fget(self),
            dtype=object)

        # Regression coefficients
        if np.any(self.switching_coeffs):
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'exog']] = [
                    '%s[%d]' % (exog_name, i) for exog_name in self.exog_names]
        else:
            param_names[self.parameters['exog']] = self.exog_names

        # Variances
        if self.switching_variance:
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'variance']] = 'sigma2[%d]' % i
        else:
            param_names[self.parameters['variance']] = 'sigma2'

        return param_names.tolist()

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.
        """
        # Inherited parameters
        constrained = super().transform_params(
            unconstrained)

        # Nothing to do for regression coefficients
        constrained[self.parameters['exog']] = (
            unconstrained[self.parameters['exog']])

        # Force variances to be positive
        constrained[self.parameters['variance']] = (
            unconstrained[self.parameters['variance']]**2)

        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        # Inherited parameters
        unconstrained = super().untransform_params(
            constrained)

        # Nothing to do for regression coefficients
        unconstrained[self.parameters['exog']] = (
            constrained[self.parameters['exog']])

        # Force variances to be positive
        unconstrained[self.parameters['variance']] = (
            constrained[self.parameters['variance']]**0.5)

        return unconstrained


class MarkovRegressionResults(markov_switching.MarkovSwitchingResults):
    r"""
    Class to hold results from fitting a Markov switching regression model

    Parameters
    ----------
    model : MarkovRegression instance
        The fitted model instance
    params : ndarray
        Fitted parameters
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    cov_type : str
        The type of covariance matrix estimator to use. Can be one of 'approx',
        'opg', 'robust', or 'none'.

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    """

    def forecast(self, steps:int=1, exog=None, method:str=None) -> np.ndarray:
        """
        Forecast the next steps of the model.

        Parameters
        ----------
        self : A MarkovRegressionResults object
        steps : int, optional
               The number of future steps where to forecast the model. Default is 1.
        exog : array-like, optional
               A [steps, k_regimes] array with the values of the exogenous variables.
        method : str, optional
                If None, It returns a tuple (a,b) of two array[steps, k_regimes]
                with a the forecasted values and b its probabilities.
                If 'MAP' the Maximum A Posteriori value is returned.
                If 'EV' the expected value.

        Returns
        -------
        forecast : array [steps] or tuple (array[steps, k_regimes], array[steps, k_regimes])
            Array or tuple according with the value of the method parameter.
        """
        if (exog is None and self.model.k_exog > 0) or (
            exog.shape[0] < steps) or (
            exog.shape[1] != self.model.k_exog):
            raise ValueError("The exog array should have shape (steps, k) with k the number of exog params.")
        if method not in (None, 'MAP', 'EV'):
            raise ValueError(f"The prediction method should be None, 'MAP' or 'EV', not {method}")

        next_states_p_l = np.zeros((steps, self.k_regimes), dtype=self.params.dtype)
        last_state_p_l = np.log(self.smoothed_marginal_probabilities.iloc[-1].values)
        #TODO: Make this work with time-varying transition matrices.
        rtm = self.model.regime_transition_matrix(self.params)
        if rtm.shape[2] != 1:
            raise ValueError('Forecast with time-varying transition matrices not implemented still.')
        transition_matrix_l = np.log(rtm.squeeze(axis=2))
        for i in range(steps):
            last_state_p_l = logsumexp(transition_matrix_l + last_state_p_l[np.newaxis,:], axis=1)
            next_states_p_l[i, :] = last_state_p_l
        next_states_p = np.exp(next_states_p_l)

        # Trend
        trend_exog = None
        if self.model.trend == 'c':
            trend_exog = np.ones((steps, 1))
        elif self.model.trend == 't':
            trend_exog = (np.arange(steps) + 1 + self.model.nobs)[:, np.newaxis]
        elif self.model.trend == 'ct':
            trend_exog = np.c_[np.ones((steps, 1)),
                               (np.arange(steps) + 1 + self.model.nobs)[:, np.newaxis]]
        if trend_exog is not None:
            exog = trend_exog if exog is None else np.c_[trend_exog, exog[:steps]]

        forecast_exog = np.zeros((steps, self.k_regimes), dtype=self.params.dtype)
        for i in range(self.k_regimes):
            forecast_exog[:, i] = np.dot(exog, self.params.iloc[self.model.parameters[i, 'exog']])

        forecast = forecast_exog
        if method == 'MAP':
            forecast = forecast[np.arange(steps), np.argmax(next_states_p, axis=1)]
        elif method == 'EV':
            forecast = np.average(forecast, axis=1, weights=next_states_p)
        else:
            return (forecast, next_states_p)

        return forecast


class MarkovRegressionResultsWrapper(
        markov_switching.MarkovSwitchingResultsWrapper):
    pass
wrap.populate_wrapper(MarkovRegressionResultsWrapper,  # noqa:E305
                      MarkovRegressionResults)
