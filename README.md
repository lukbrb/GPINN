# GPINN

## 1. Introduction

In the study of galaxy dynamics, for computing the total potential of the galaxy, one could in principle add the
point-mass potentials of all the stars together; as much of the mass of the galaxy resides in the stars.
A typical galaxy accounts for approximately $10^{11}$ stars, making the last solution impracticable [2]. For the purpose of
galactic dynamical modeling, it is generally sufficient to properly describe the galaxy density field and compute its
gravitational potential through the Poisson equation.  

As often in physics, there exist some analytical solutions -here analytic potential-density pairs for simple systems,
 but in more realistic situations, numerical quadrature is required [3]. Numerical solutions are often time-consuming
 to compute, especially when dealing with dynamical modeling of real galaxies and Bayesian parameter
 estimation methods [6].

Artificial intelligence (AI) has become an essential part of modern scientific research in numerous fields.
This omnipresence is mostly justified by the enormous amount of data now available. There exists however some fields
where the quantity of data is limited. In such cases, one would like to be able to use AI techniques to use the
available data.  
An ingenious solution to the latter problem has been brought by [1]. The idea is first to use the properties of neural
networks as universal function approximators, and to take advantage of their auto-differentiation property. We then use
the knowledge we have from physics (symmetry, invariance, or conservation principles originating from the physical laws
that govern the observed data), to restrain the parameter space during the neural network training phase. We call these
"Physics-Informed Neural Networks” (PINNs). They can be used to solve nonlinear partial differential equations (PDEs)
encountered in various physics problems - such as advection problems, diffusion problems, flow problems etc...
This PINN construction can tackle a wide range of problems in computational sciences and, leads to the introduction of
new classes of numerical solvers for partial differential equations.  

As in grid-based simulations with adaptive mesh refinement solving the Poisson equation via the multigrid approach, or
Fourier methods, is quite expensive, PINNs could introduce a change of paradigm in the studies of complex systems by
pushing forward hardware and computational limitations.

## 2. Specific problems addressed in the thesis

Using the properties of neural networks for fitting data, and their efficiency for solving PDEs, the main aim of this
thesis is to improve galaxy modeling. Specific density profiles will be studied. Starting from simple analytic density
profiles, the thesis will extend to more complicated axisymmetric density profiles, closer to reality. This work could
allow researchers working on galaxy dynamics to do parameter estimation with a dramatic gain in computational time and
accuracy, opening previously impracticable routes. Other applications of PINNs in astrophysics exist; in the same vein
they could be used to speed up initial conditions creation for galaxy simulations. Complex system simulations could
also be possible.

We make use of the recent development of PINNs for solving Poisson’s equation [5], for the gravitational potential.
As a proof of concept, Hernquist density profile is first tested. In this case, the gravitational potential can be
computed analytically and may be used for testing and validating the accuracy of the PINN. Once the ability of PINNs
to solve Poisson’s equation is attested, we try to solve for the potential for the Denhen density profile. While the
Hernquist profile has two parameters, mass and scale-radius, the Denhen profile has three. The mass and scale radius as
well as the inner slope, denoted 𝛾. The Dehnen profile has an analytic solution, but the Poisson equation now depends
on a further parameter 𝛾. With this second profile, we want to assess the accuracy of PINNs in solving parametric
differential equations.

The JASMIN [3] code is used to verify our results. Finally, we continue with more realistic axisymmetric density
profiles, such as the Miyamoto and Nagai and, the more realistic exponential thick disc. Depending on the success of the
latter, we might expand the project by applying the same technique for solving the Jeans equation of gravity, commonly
used for predicting galaxy velocity fields.  

From the right choice of architecture for the network, to clever tricks for attaining convergence, or simplifying the
equations, the difficulties encountered in this project are plural. The choice of architecture for physics-informed
neural networks is still an on-going research topic, and might be the most demanding task. It is also worth mentioning
that to our knowledge, no work has been done on solving the gravitational Poisson equation, nor the Jeans equations,
with PINNs.

## 3. References

[1]. Raissi et al. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
involving nonlinear partial differential equations” (<https://doi.org/10.1016/j.jcp.2018.10.045>)

[2]. Binney et al. “Galaxy Dynamics”
(<http://www.tevza.org/home/course/AF2016/books/Galactic%20Dynamics>,%20James%20Binne y%20(2ed.,%20).pdf)

[3]. Caravita et al. “Jeans modeling of axisymmetric galaxies with multiple stellar populations
“ (<https://arxiv.org/pdf/2102.09440.pdf>)

[4]. Cuomo, S., Di Cola, V.S., Giampaolo, F. et al. Scientific Machine Learning Through Physics–Informed Neural Networks:
Where we are and What’s Next. J Sci Comput 92, 88 (2022) (<https://doi.org/10.1007/s10915-022-01939-z>)

[5]. Kharazmi et al. “Variational physics-informed neural networks for solving partial differential equations”
(<https://arxiv.org/abs/1912.00873>)

[6]. Rigamonti, Fabio, et al. "Maximally informed Bayesian modelling of disc galaxies." Monthly Notices of the Royal
Astronomical Society 513.4 (2022): 6111-6124.
