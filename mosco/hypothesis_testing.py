import numpy as np
from scipy import stats


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


def scipy_ttest_ind_from_stats(mu_1: float, mu_2: float, sigma_1: float, sigma_2: float, n_1: int, n_2: int, equal_var: bool = False):
    """A helper function that takes statistics as arguments and calls scipy.stats.ttest_ind_from_stats to perform independent two-sample t-test. 

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
    equal_var : bool, optional
        True if the variances of the two groups are equal, by default False. 

    Returns
    -------
    tstat : float
        The estimated t-statistic. 
    p_value : float 
        The p-value. 

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html

    """

    tstat, p_value = stats.ttest_ind_from_stats(
        mean1=mu_1, std1=sigma_1, nobs1=n_1,
        mean2=mu_2, std2=sigma_2, nobs2=n_2,
        equal_var=equal_var
    )
    return tstat, p_value
