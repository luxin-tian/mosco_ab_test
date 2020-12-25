### Two-sample t-test 

Based on the Central Limit Thereom, the sample mean of a series of independent Bernoulli random variable has a student's _t_ distribution, which asymptotically converges to normality. 

We perform [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t_test) with the assumption that the two population have equal or unequal sample sizes and unequal variances. The _t_ statistic is constructed as follows: 

$$t = \frac{\bar{X_1} - \bar{X_2}}{s_{\bar{\delta}}}$$

where 

$$s_{\bar{\delta}} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$

The distribution of the constructed _t_ statistic is approximated as an ordinary Student's t-distribution with the degrees of freedom given by the Welch-Satterthwaite equation: 

$$d.f. = \frac{(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2})^2}{\frac{(\frac{s_1^2}{n_1}^2)}{n_1} + \frac{(\frac{s_2^2}{n_2}^2)}{n_2}}$$ 

