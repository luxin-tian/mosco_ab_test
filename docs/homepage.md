## What is A/B Test?

> A/B testing (also known as bucket testing or split-run testing) is a user experience research methodology. A/B tests consist of a randomized experiment with two variants, A and B. It includes application of statistical hypothesis testing or "two-sample hypothesis testing" as used in the field of statistics. A/B testing is a way to compare two versions of a single variable, typically by testing a subject's response to variant A against variant B, and determining which of the two variants is more effective.

Read more on [Wikipedia](https://en.wikipedia.org/wiki/A/B_testing). 

## What does Mosco do? 

Mosco is an A/B test dashboards that provides simple tools for causal inference and hypothesis testing. We have implemented the following functionailities: 

- Student's t-test 
  - Bernoulli variable
  - Numeric variable
- ...

## Python API 

Mosco provides a Python API for all the implemented hypothesis testings. Read more on the [documentation](#). 

可以把所有的functions封装成为一个Python module，可以使用`Sphinx`基于各个function中的docstring来一键生成documentation，然后你的软件包就可以由其他用户import在Python中使用啦。

### Acknowledgement 

Developed by [@Yezi_Li](mailto:yezili@link.cuhk.edu.cn) with enthusiam at [the Chinese University Of Hong Kong, Shenzhen](https://www.cuhk.edu.cn)'. 

This is the final project of ECOXXXX course. The data app is rendered by [Streamlit.io](https://www.streamlit.io/). 
