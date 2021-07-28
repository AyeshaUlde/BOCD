# BOCD

Python implementation of [Bayesian Online Changepoint Detection](https://arxiv.org/pdf/0710.3742.pdf) for a Normal-Gamma model with unknown mean and variance parameters. 

BOCD.py is a python script that estimates the run-length posterior distribution <img src="https://render.githubusercontent.com/render/math?math={\large p(r_{t}| \textbf{x}_{1:t})}"> of the [well-drilling nuclear magnetic response data](https://github.com/AyeshaUlde/BOCD/blob/main/NMRlogWell.mat) using the Algorithm 1 mentioned in [Bayesian Online Changepoint Detection](https://arxiv.org/pdf/0710.3742.pdf).

Some useful resources for understanding and implementing the mentioned paper are:
- [Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
- http://gregorygundersen.com/blog/2019/08/13/bocd/
- http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/


### New to Bayesian Statistics?

Check out some of these resources:
- [Bayes Theorem explained by 3Blue1Brown](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [Bayesian vs Frequentist Probability](https://www.youtube.com/watch?v=YsJ4W1k0hUg)
- [Beginner's guide to probability density](https://www.youtube.com/watch?v=ZA4JkHKZM50)
- [Understanding prior and posterior predictive distributions](https://www.youtube.com/watch?v=R9NQY2Hyl14&pp=sAQA)
