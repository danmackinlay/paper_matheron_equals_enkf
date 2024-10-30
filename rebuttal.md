## Reviewer 1 fxbi

Thanks for the productive, fair and useful review!

Thanks for the productive, fair and useful review!

Question: The linearization is mentioned as an issue, would similar tricks as
used in unscented Kalman filters, to improve over a standard EKF, be applicable here as well?

Answer: We believe this hypothesis is likely to be correct; there is probably a way that techniques from Unscented Kalman filtering could be applied to this problem.
However, we also believe that it is not likely to be straightforward in the high-dimensional setting that this paper targets.
In typical application, the number of sigma points, say $M$, must equal or exceed the latent dimension $D\_z$ of the problem; commonly $M=2D\_z+1$, involving storing a $M \times D\_z$ matrix, which is at least as large as the $D\_z \times D\_z$ matrix that we were unwilling to store in the problem setup.
More generally, discovering how to use the unscented transform in this context, possibly via a dimension reduction, would be potentially fruitful. It is, in our opinion, nonetheless a whole non-trivial paper in its own right.We propose to address this by changing l33 RHS 

>This generates closed-form ensemble update rules which are capable of assimilating multiple observations without ever constructing the prior covariance matrix, and thus scales to very high dimensions.

 appending 

>Unlike other sample-based approximations such as Unscented Kalman filters, the number of samples needed grows sub-linearly with dimension $D\_z$ under the assumption that the model is well-approximated by a low-rank distribution.


_Question : In the introduction the paper mentions that the method approximates observations as Gaussian variates. How limiting is this assumption for real-world problems and how does it impact solution quality when this assumption is not met?_

Yes, this is a tough one.
The assumption of Gaussianity in the field of Ensemble Kalman-type techniques is delicate, and an active area of research.
We are not aware of any results that answer this analytically in the general case; in practice approaches such the one use, which test how well the model performed empirically, seem to be the most effective.
The model as given is clearly non Gaussian, since the Navier-Stokes solutions are not linear in the values of either the density of velocity components of the state vector.
However, as such the method is identical to most attempts to invert high-dimensional models, in that the only reliable verification is empirical testing on similar problem.
The method here does not improve upon this, but nor is it worse than competing methods.
We have addressed this point by amending the poposoed "Limitations" — see below.

Limitations:
This is not the only review to request limitations be mentioned separately. Accordingly we propose that the revised paper contain a section as follows:

LIMITATIONS AND ADVERSE SOCIETAL IMPACTS

The method as described here makes assumptions of Gaussianity upon all variates and linearity upon the operator $\mathcal{P}$ which are not satisfied in practice, and there is no known method to quantify the additional error introduced by this assumption in general.
The effectiveness of the method in the desire domain must be verified empirically by the user.
There are no obvious adverse societal impacts to improving the scope of inverse problem solutions in particular, apart from addition to the usual dangers of bringing yet more previously intractable problems into the reach of computational analysis.

## Reviewer 2 3T1w

Weakness:

1. The novelty of the proposed method is unclear and seems to be minimal. IIUC, the proposed method applies
   existing techniques (ensemble approximation, Matheron updates) to model inversion, while from the last
    paragraph of the intro, it seems Ensemble Kalman filter has already applied similar techniques to dynamic
   systems to discover states which is a harder task.

Yes, this is a central point.

On one hand, from the perspective of Ensemble Kalman filtering, this might be regarded "just" a special case of the classical Ensemble Kalman filter of Evensen 1994  with the process dynamics set to be trivial, i.e. constant, and the observation noise set to be i.i.d. Gaussian. 
On the other hand, that supposedly special case, does not seem to be used in the literature for solving this class of problem.
Further, the connection between the Ensemble Kalman update and the Matheron update rule, which is currently a subject of intense research in the Machine Learning community, appears not to be understood.
Connecting these two is of interest, suggesting techniques may be cross-propagated across these two bodies of literature.

How novel/publishable any of these insights should be is "in the eye of the beholder".
Is the research "foundational", or  "translational"?

Proposed remediation:


We propose to clarify this point by adding the following text to the paper 

l33 RHS:

>The method is structurally similar to the Ensemble Kalman filter, in that all updates between random variates are conducted via ensembles of samples rather than explicit densities. However, METHO exploits the closed form Matheron updates for static parameters rather than states. This generates closed-form ensemble update rules which are capable of assimilating multiple observations without ever constructing the prior covariance matrix, and thus scales to very high dimensions.

→

>The method is structurally similar to the Ensemble Kalman filter, in that all updates between random variates are conducted via ensembles of samples rather than explicit densities.  However, METHO applies the sample-wise updates to static parameters rather than states, which can be interpreted either as an ensemble variant of the Matheron rule, or a static parameter version of the ensemble Kalman filter. This generates closed-form ensemble update rules which are capable of assimilating multiple observations without ever constructing the prior covariance matrix, and thus scales to very high dimensions.


**I hope the authors could discuss more the assumption of having a forward operator: wh en we do have access to it, if not, how we can approximate it, etc.**

**The assumption of having a forward operator only holds for limited scenarios.**

Indeed the requirement that there be a forward operator is a restriction of a sort.
However, we assert that this as assumption is in fact broader and the class of problems that may be handled thereby is powerful that it might initially seem.
There are three points we should address:

**1.** We assert that model inversion in high dimension, even with a precisely known forward operator is a generally unsolved problem of major practical interest in application areas, as the ongoing rate of publications addressing it will attest. We refer the reviewers to section 2.2 for a brief summary of some activity in this area. The task is, moreover, of great interest in practice. Many domain experts rely upon "gold standard" forward simulators but lack model inversion techniques, for example MODSIM in groundwater hydrology, SPARK in wildfire simulation, and so on. 

Proposed remediation: l059 LHS insert the following text:
After 

>Model inversion has a long history of methodological development in the geophysical sciences, where problems often involve high-dimensional parameter spaces and complex physical models.

Add

> High dimensional models with well-understood forward operators but poorly-developed inversion techniques are ubiquitous in industry; for example MODFLOW in hydrology [ref], APSIM in agricultural simulation [ref], fluid flow simulators in rocketry [Gramacy et al 2008], Global Circulation Models in atmospheric physics and so on.

* The forward operator that we know here is a _stochastic_ forward operator, which means that $\mathscr{P}$ operator is known only up to an irreducible aleatoric noise term. This generally regarded as a challenging case for model inversion. The case of _stochastic_ models, with high-dimensional, nonlinear, irreducible aleatoric uncertainty, comprises a particularly challenging sub-category within this family. For example, it is not easily handled by the naïve linearised Gaussian technique and would require a far more-burdensome stochastic Itô-Taylor expansion even if it scaled up to the size of our experiment. This is not infeasible, but seems to be generally regarded as too tedious to be worthwhile. We propose the following update to clarify this by appending l25RHS:

>Moreover, METHO can handle cases where the forwrad operator is subject to high-dimensional aleatoric noise, which is generally regarded as a particularly challenging case.

* The degree to which we need to "know" the operator is looser than it might seem. While we talk about a "latent forcing" with a known operator, in fact the treatment of the latent parameter is completely generic. If the forward operator is a parametric one with unknown parameters $\theta$, we can include those unknown parameters in the latent forcing vector concatenating $\theta$ and $\mathbf{u}$ . The inference then will still be well-posed and it may be accurate, so long as the unknown parameters of the parametric operator as "sufficiently jointly Gaussian" with regard to the forward prediction. We propose the following update to clarify this, appending at l117LHS:

> The time invariant parameter $\mathbf{u}$ in the examples here is a latent spatial field, but it could, alternatively or additionally, include unobserved parameters of the operator itself, such as viscosity, density parameters.

Indeed the requirement that there be a forward operator is a restriction of a sort.
However, we assert that this as assumption is in fact broader and the class of problems that may be handled thereby is powerful that it might initially seem.
There are three points we should address:

**1.** We assert that model inversion in high dimension, even with a precisely known forward operator is a generally unsolved problem of major practical interest in application areas, as the ongoing rate of publications addressing it will attest. We refer the reviewers to section 2.2 for a brief summary of some activity in this area. The task is, moreover, of great interest in practice. Many domain experts rely upon "gold standard" forward simulators but lack model inversion techniques, for example MODSIM in groundwater hydrology, SPARK in wildfire simulation, and so on. 

Proposed remediation: l059 LHS insert the following text:
After 

>Model inversion has a long history of methodological development in the geophysical sciences, where problems often involve high-dimensional parameter spaces and complex physical models.

Add

> High dimensional models with well-understood forward operators but poorly-developed inversion techniques are ubiquitous in industry; for example MODFLOW in hydrology [ref], APSIM in agricultural simulation [ref], fluid flow simulators in rocketry [Gramacy et al 2008], Global Circulation Models in atmospheric physics and so on.

* The forward operator that we know here is a _stochastic_ forward operator, which means that $\mathscr{P}$ operator is known only up to an irreducible aleatoric noise term. This generally regarded as a challenging case for model inversion. The case of _stochastic_ models, with high-dimensional, nonlinear, irreducible aleatoric uncertainty, comprises a particularly challenging sub-category within this family. For example, it is not easily handled by the naïve linearised Gaussian technique and would require a far more-burdensome stochastic Itô-Taylor expansion even if it scaled up to the size of our experiment. This is not infeasible, but seems to be generally regarded as too tedious to be worthwhile. We propose the following update to clarify this by appending l25RHS:

>Moreover, METHO can handle cases where the forward operator is subject to high-dimensional aleatoric noise, which is generally regarded as a particularly challenging case.

* The degree to which we need to "know" the operator is looser than it might seem. While we talk about a "latent forcing" with a known operator, in fact the treatment of the latent parameter is completely generic. If the forward operator is a parametric one with unknown parameters $\theta$, we can include those unknown parameters in the latent forcing vector concatenating $\theta$ and $\mathbf{u}$ . The inference then will still be well-posed and it may be accurate, so long as the unknown parameters of the parametric operator as "sufficiently jointly Gaussian" with regard to the forward prediction. We propose the following update to clarify this, appending at l117LHS:

> The time invariant parameter $\mathbf{u}$ in the examples here is a latent spatial field, but it could, alternatively or additionally, include unobserved parameters of the operator itself, such as viscosity, density parameters.

3. The experiment evaluations are limited, only conducted on two simulated cases, with linearised Gaussian Process as the only baseline.

**2. For the experiment datasets, I wonder if the authors could test on more realistic (or real-world) datasets, or
   discuss how the used Navier-stoke dataset is challenging to make the results more convincing.**

Thanks for raising this! We should clarify that the dataset is not a dataset *per se*, but that training examples are generated on the fly from a simulator.
This may sounds contrived, but is in practice common — the examples in the introduction concerning PEST and GLUE are examples of precisely this type.
Frequently inverse problems suffer from a lack of *real* datasets suitable for supervised training; In hydrogeology, for example, there will never be a ground truth observation unless we are prepared to excavate and instrument over w hole catchment.
In this setting, the use of simulators is a strong way to incorporate physical constraints into reasoning about the system.

That said, the point is well-taken that we need to connect these simulation-based inputs more closely to real-world problems, so we should include an experiment which shows the breakdown of the methods under consideration as the dataset grows more challenging.

Our proposed remediation is to plot results for two additional problems

1. a problem that is very-nearly-linear  (pure advection-diffusion), and a
2. a highly nonlinear problem (very turbulent flow).

We would expect this would demonstrate the strengths and weaknesses of the problem and baseline on successively more challenging data, whilst remaining comparable.

As for the baseline _inference methods,_ we can in principle test against additional  approaches, such as optimisation, and in fact we did this during the early development of this paper.
We did not include these results because the comparison with optimisation is complicated and we feel detracts from the main message without providing much extra information.
The difficulty is that comparison is not "apples-to-apples", i.e. extracting a posterior estimate by optimisation is not well-defined, and the posterior uncertainty is central to this work.
We could introduce additional tools to estimate posterior coverage (e.g. Laplace approximations) but then we have once again recovered an intractably large posterior covariance matrix, which is nearly identical to that induced by the linearised Gaussian method, so we have the same difficulty
To put this another way, when we modify optimisation to be comparable, we are in effect once again solving an (almost) a linearised Gaussian inversion.

We hope to emphasise that our goal in this paper is *not* to achieve SOTA results on any given benchmark, but rather to

* demonstrate a method which produces regularised, computationally-affordable posterior estimates for a  high-dimensional, static parameter, which is a recognised challenge without, as far as we know, any method as simple as the one outlined here.
* incidentally connect the body of research from the Gaussian Process community on the Matheron update and the Ensemble Kalman Smoother-type techniques from the Data Assimilation community

Suggested remediation: Amend section 4.1:

>...towards a priori plausible solutions, and assigning a posterior weight to various candidate solutions. In the pure optimisation setting, we can also use regularisation to penalise the parameter-space towards plausible solutions, although inclusion of the regularisation term loses the intuitive interpretation that we gain from employing a prior.

→

>... towards a priori plausible solutions. In a generalized least squares setting optimisation setting this regularised becomes equivalent to MAP estimation in a Bayesian linear model (see e.g. Bickson 2009 Ch 1). Accordingly we do not pursue optimisation approaches further here.

Would that address the concerns? If we wanted to go deeper, we could recover the optimisation-based solution from that, e.g. by arguing that Ordinary Least Squares is the special case with trivial regularisation penalty and isotropic covariance.


3. For other baselines, though authors mention conceptually in Sec 4.1 why optimization methods may have
   worse performance than the proposed method (e.g., point estimate doesn't take uncertainty into
   consideration), I wonder if it's possible to empirically show that. After all, Sec 5.1 is evaluated on a deterministic
   system only. Or if other methods need extra assumptions, e.g., access to likelihood, it would be more
   compelling to see that your method achieves similar performance without those assumptions)


## Reviewer 2 KT5D

On originality
I am not familiar enough with the data assimilation literature, but from my understanding the ensemble Kalman
filter (EnKF) is a very commonly used method there, and the proposed method seems very closely related to this
method. Vaguely speaking, it almost seems like the method is actually an EnKF. It would therefore be very helpful to
discuss similarities and differences between "METHO" and commonly used EnKF methods in more detail, to clarify
such questions and help readers situate the paper better.
On the evaluation
METHO is compared only to the "linearised Gaussian method", which to me seems to be (at the very least closely
related to) an extended Kalman filter (EKF). In an ideal comparison, both methods aim to compute the same
quantity, and we could then compare how well they approximate this quantity, and how much computational effort
was necessary. But since they differ in their use of , their probabilistic model differs, and they compute different
posteriors. I personally see two improvements that would, from my perspective, significantly strengthen the
experiments:
1. Incude results of METHO for , which could then be compared to the EKF.
2. Implement the EKF in square-root form for improved numerical stability. Then, much lower should be
possible without running into numerical issues. Note that the square-root EKF typically requires some QR
decompositions, which hurt its the runtime (and thus in comparison, METHO might have an advantage)
Generally speaking, I also believe that having more quantitative evaluations would be very beneficial; right now
only Figure 4 compares the proposed method to an alternative. This includes both having more different
experimental setups, but maybe more importantly also comparing to other methods (such as those EnKF
approaches commonly used in data assimilation; if applicable).
Questions:
As mentioned above, my main concern is with the relation of METHO to the ensemble Kalman filtering methods
used in data assimilation; a clarification by the authors on their similarities and differences would be very helpful.
On 4.1 Inversion by optimization: The end of the second paragraph discusses the lack of uncertainties in MAP
estimates. I believe that "Laplace approximations" should be at least mentioned in this context, as they target
exactly this setup: First computing a MAP with optimization, and then forming a Gaussian approximate posterior by
essentially just computing Hessians.
Other minor comments:
• Equation 5: I believe w and w should be flipped; see Wilson 2021, Eq 4.
• L. 191: "(IS))"
• Eq. 16 introduces and , which I think were not introduced.
• "This cost [ ] is favourable in comparison with ": The last part is probably a typo and should probably be " ".

Limitations:
There does not seem to be a dedicated "limitations" section in the paper, and I could not really find much discussion on drawbacks of the proposed method; the limited empirical evaluation also does not make it easy for readers to
assess these. More information on this would be helpful.

