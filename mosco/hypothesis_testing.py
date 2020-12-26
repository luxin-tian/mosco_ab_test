import numpy as np
import scipy.stats
import statsmodels.stats.power


def bernoulli_stats(visitors_1: int, visitors_2: int, conversions_1: int, conversions_2: int):
    """Calculate the means and standard deviations of two sample groups that have Bernoulli distributions. 

    Parameters
    ----------
    visitors_1 : int
        The number of total visitors of sample group 1. 
    visitors_2 : int
        The number of total visitors of sample group 1. 
    conversions_1 : int
        The number of conversions of sample group 1. 
    conversions_2 : int
        The number of conversions of sample group 2. 

    Returns
    -------
    mu_1 : float 
        The sample mean of group 1. 
    mu_2 : float 
        The sample mean of group 2. 
    sigma_1 : float 
        The sample standard deviation of group 1. 
    sigma_2 : float 
        The sample standard deviation of group 2. 

    """
    mu_1 = conversions_1 / visitors_1
    mu_2 = conversions_2 / visitors_2
    sigma_1 = np.sqrt(mu_1 * (1 - mu_1))
    sigma_2 = np.sqrt(mu_2 * (1 - mu_2))
    return mu_1, mu_2, sigma_1, sigma_2


def scipy_ttest_ind_from_stats(mu_1: float, mu_2: float, sigma_1: float, sigma_2: float, n_1: int, n_2: int):
    """A helper function that wraps scipy.stats.ttest_ind_from_stats [1] to perform independent two-sample t-test. 

    Parameters
    ----------
    mu_1 : float
        The sample mean of group 1. 
    mu_2 : float
        The sample mean of group 2. 
    sigma_1 : float
        The sample standard deviation of group 1. 
    sigma_2 : float
        The sample standard deviation of group 2. 
    n_1 : int
        The sample size of group 1. 
    n_2 : int
        The sample size of group 2. 

    Returns
    -------
    tstat : float
        The estimated t-statistic. 
    p_value : float 
        The two-tailed p-value. 
    tstat_denom : float 
        The denominator of the t-statistic. 
    pooled_sd : float 
        The pooled standard deviation. 
    effect_size : float 
        The standardized effect size, calculated using Cohen's d. 

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html

    """
    tstat, p_value = scipy.stats.ttest_ind_from_stats(
        mean1=mu_1, std1=sigma_1, nobs1=n_1,
        mean2=mu_2, std2=sigma_2, nobs2=n_2,
        equal_var=False 
    )
    pooled_sd = np.sqrt(((sigma_1 ** 2 * (n_1 - 1)) + (sigma_2 ** 2 * (n_2 - 1))) / (n_1 + n_2 - 2))
    effect_size = (mu_1 - mu_2) / pooled_sd 
    tstat_denom = (mu_1 - mu_2) / tstat
    return tstat, p_value, tstat_denom, pooled_sd, effect_size


def sm_tt_ind_solve_power(effect_size=None, n1=None, n2=None, alpha=None, power=None, ratio=None, hypo_type='Two-sided', if_plot=True):
    """A helper function that wraps scipy.stats.ttest_ind_from_stats [1] to perform power analysis for two-sample t-test. 
    Exactly one of the following needs to be None, all others need numeric values: effect_size, n1 and n2, alpha, and power. 

    Parameters
    ----------
    effect_size : float
        The standardized effect size, mean divided by the standard deviation.
    n1 : int or float
        The sample size, number of observations.
    alpha : float in interval (0,1)
        The significance level, e.g. 0.05, is the probability of a type I
        error.
    power : float in interval (0,1)
        The power of the test, or beta, e.g. 0.8, is one minus the probability of a
        type II error. 
    hypo_type : str, 'two-sided' (default) or 'one-sided'
        The type of the alternative hypothesis, indicating whether the power is calculated for a
        two-sided (default) or one sided test.

    Returns
    -------
    value : float or (float, matplotlib.figure.Figure)
        If if_plot is False then returns the value of the parameter that was set to None in the call, 
        which solves the power equation given the remaining parameters. If if_plot is True, also returns the Figure instance. 

    References
    ----------
    .. [1] https://www.statsmodels.org/stable/generated/statsmodels.stats.power.tt_ind_solve_power.html#statsmodels.stats.power.tt_ind_solve_power

    """
    if n1 is not None and n2 is not None and ratio is None:
        ratio = n2 / n1
    ttest_ind_power = statsmodels.stats.power.TTestIndPower()
    hypo_type = {'Two-sided': 'two-sided', 'One-sided': 'larger'}[hypo_type]
    value = ttest_ind_power.solve_power(
        effect_size=effect_size, nobs1=n1, alpha=alpha, power=power,
        ratio=ratio, alternative=hypo_type
    )

    if if_plot: 
        effect_size_candidates = [0.05, 0.1, 0.2, 0.5, 0.8]
        if effect_size is not None and effect_size not in effect_size_candidates: 
            effect_size_candidates.append(effect_size)
        if n1 is None: 
            nobs_ub = value
        else: 
            nobs_ub = max(100, n1)
        fig = ttest_ind_power.plot_power(dep_var='nobs', nobs=np.linspace(5, nobs_ub), 
            effect_size=np.array(sorted(effect_size_candidates)), 
            alpha=0.05, ax=None, title='Power Analysis'
        )
        return value, fig
    else: 
        return value