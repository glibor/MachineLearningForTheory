# Using Machine Learning to improve theories (based on Fudenberg and Liang, 2020)
<hr> 

This program is a Python implementation of theory and (some of the) applications developed in the paper
***Machine Learning for Evaluating and Improving Theories*** by Fudenberg and Liang, 
available [here](https://economics.mit.edu/sites/default/files/2022-08/Machine%20Learning%20for%20Evaluating.pdf).

The  original paper's main message is summarized in its conclusion:

<blockquote>
[...] machine learning and associated algorithmic techniques can aid
in the improvement of economic theories. When theories are incomplete, machine
learning can help researchers identify regularities that are not captured by existing
models and then develop new theories that predict better. Conversely, when a
theory is highly complete, algorithmic techniques can show whether this is simply
due to the theory’s ability to fit any possible data, or whether the good fit results
from the theory describing behaviors in the real world
</blockquote>


The files include the database from [Bruhnin et al, 2010](https://www.econometricsociety.org/publications/econometrica/2010/07/01/risk-and-rationality-uncovering-heterogeneity-probability)
and the main script. Essentially the intent is to replicate the following table:

![Paper table](/assets/images/paper_results.PNG "Paper results. Source:Fudenberg and Liang, 2020")

The models are trained in the following way: the data is divided in ten parts and 
the parameter estimation is conducted using a _least squares estimator_ in nine of them. The remaining data
is used as out-of-sample test.

Running `main.py` estimates the certain equivalent models and print the results, as they were displayed in the paper.

<h3> References </h3>

<hr>

**Fudenberg, Drew, and Liang, Annie; 2020.**
_Machine Learning for Evaluating and Improving Theories_. ACM SIGecom Exchanges: Vol. 18, No. 1, 4-11.





**Bruhin, A., Fehr-Duda, H., and Epper, T 2010.** 
_Risk and rationality: Uncovering heterogeneity in probability distortion_. Econometrica 78, 4, 1375--1412




**Fudenberg, Drew et al. 2019.** 
_Measuring the Completeness of Theories_. 
PIER Working Paper.



**Fudenberg, Drew; Gao, Wayne; and Liang, Annie; 2020.** 
_How Flexible is that Functional Form? Quantifying the Restrictiveness of Theories_. arXiv preprint arXiv:2007.09213
