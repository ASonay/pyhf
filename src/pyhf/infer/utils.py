"""Inference for Statistical Models."""

from .. import get_backend
from .calculators import AsymptoticCalculator, ToyCalculator


def create_calculator(calctype, *args, **kwargs):
    """
    Creates a calculator object of the specified `calctype`.

    See :py:class:`~pyhf.infer.calculators.AsymptoticCalculator` and :py:class:`~pyhf.infer.calculators.ToyCalculator` on additional arguments to be specified.

    Example:

        >>> import pyhf
        >>> import numpy.random as random
        >>> random.seed(0)
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0],
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> mu_test = 1.0
        >>> toy_calculator = pyhf.infer.utils.create_calculator(
        ...     "toybased", data, model, ntoys=100, track_progress=False
        ... )
        >>> qmu_sig, qmu_bkg = toy_calculator.distributions(mu_test)
        >>> qmu_sig.pvalue(mu_test), qmu_bkg.pvalue(mu_test)
        (0.14, 0.76)

    Args:
        calctype (`str`): The calculator to create. Choose either 'asymptotics' or 'toybased'.

    Returns:
        calculator (`object`): A calculator.
    """
    return {'asymptotics': AsymptoticCalculator, 'toybased': ToyCalculator,}[calctype](
        *args, **kwargs
    )


def hypotest(
    poi_test,
    data,
    pdf,
    init_pars=None,
    par_bounds=None,
    calctype='asymptotics',
    **kwargs,
):
    r"""
    Compute :math:`p`-values and test statistics for a single value of the parameter of interest.

    See :py:class:`~pyhf.infer.calculators.AsymptoticCalculator` and :py:class:`~pyhf.infer.calculators.ToyCalculator` on additional keyword arguments to be specified.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> mu_test = 1.0
        >>> CLs_obs, CLs_exp_band = pyhf.infer.hypotest(
        ...     mu_test, data, model, return_expected_set=True
        ... )
        >>> CLs_obs
        array([0.05290116])
        >>> CLs_exp_band
        array([[0.00260641],
               [0.01382066],
               [0.06445521],
               [0.23526104],
               [0.57304182]])

    Args:
        poi_test (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        pdf (~pyhf.pdf.Model): The HistFactory statistical model
        init_pars (Array or Tensor): The initial parameter values to be used for minimization
        par_bounds (Array or Tensor): The parameter value bounds to be used for minimization
        calctype (`str`): The calculator to create. Choose either 'asymptotics' (default) or 'toybased'.

    Keyword Args:
        return_tail_probs (`bool`): Bool for returning :math:`\textrm{CL}_{s+b}` and :math:`\textrm{CL}_{b}`
        return_expected (`bool`): Bool for returning :math:`\textrm{CL}_{\textrm{exp}}`
        return_expected_set (`bool`): Bool for returning the :math:`(-2,-1,0,1,2)\sigma` :math:`\textrm{CL}_{\textrm{exp}}` --- the "Brazil band"

    Returns:
        Tuple of Floats and lists of Floats:

            - :math:`\textrm{CL}_{s}`: The :math:`p`-value compared to the given threshold :math:`\alpha`, typically taken to be :math:`0.05`, defined in :xref:`arXiv:1007.1727` as

            .. math::

                \textrm{CL}_{s} = \frac{\textrm{CL}_{s+b}}{\textrm{CL}_{b}} = \frac{p_{s+b}}{1-p_{b}}

            to protect against excluding signal models in which there is little sensitivity. In the case that :math:`\textrm{CL}_{s} \leq \alpha` the given signal model is excluded.

            - :math:`\left[\textrm{CL}_{s+b}, \textrm{CL}_{b}\right]`: The signal + background :math:`p`-value and 1 minus the background only :math:`p`-value as defined in Equations (75) and (76) of :xref:`arXiv:1007.1727`

            .. math::

                \textrm{CL}_{s+b} = p_{s+b} = \int\limits_{q_{\textrm{obs}}}^{\infty} f\left(q\,\middle|s+b\right)\,dq = 1 - \Phi\left(\frac{q_{\textrm{obs}} + 1/\sigma_{s+b}^{2}}{2/\sigma_{s+b}}\right)

            .. math::

                \textrm{CL}_{b} = 1- p_{b} = 1 - \int\limits_{-\infty}^{q_{\textrm{obs}}} f\left(q\,\middle|b\right)\,dq = 1 - \Phi\left(\frac{q_{\textrm{obs}} - 1/\sigma_{b}^{2}}{2/\sigma_{b}}\right)

            with Equations (73) and (74) for the mean

            .. math::

                E\left[q\right] = \frac{1 - 2\mu}{\sigma^{2}}

            and variance

            .. math::

                V\left[q\right] = \frac{4}{\sigma^{2}}

            of the test statistic :math:`q` under the background only and and signal + background hypotheses. Only returned when ``return_tail_probs`` is ``True``.

            - :math:`\textrm{CL}_{s,\textrm{exp}}`: The expected :math:`\textrm{CL}_{s}` value corresponding to the test statistic under the background only hypothesis :math:`\left(\mu=0\right)`. Only returned when ``return_expected`` is ``True``.

            - :math:`\textrm{CL}_{s,\textrm{exp}}` band: The set of expected :math:`\textrm{CL}_{s}` values corresponding to the median significance of variations of the signal strength from the background only hypothesis :math:`\left(\mu=0\right)` at :math:`(-2,-1,0,1,2)\sigma`. That is, the :math:`p`-values that satisfy Equation (89) of :xref:`arXiv:1007.1727`

            .. math::

                \textrm{band}_{N\sigma} = \mu' + \sigma\,\Phi^{-1}\left(1-\alpha\right) \pm N\sigma

            for :math:`\mu'=0` and :math:`N \in \left\{-2, -1, 0, 1, 2\right\}`. These values define the boundaries of an uncertainty band sometimes referred to as the "Brazil band". Only returned when ``return_expected_set`` is ``True``.

    """
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    return_tail_probs = kwargs.pop('return_tail_probs', False)
    return_expected_set = kwargs.pop('return_expected_set', False)
    return_expected = kwargs.pop('return_expected', False)

    tensorlib, _ = get_backend()

    calc = create_calculator(calctype, data, pdf, init_pars, par_bounds, **kwargs)
    teststat = calc.teststatistic(poi_test)
    sig_plus_bkg_distribution, b_only_distribution = calc.distributions(poi_test)

    CLsb = sig_plus_bkg_distribution.pvalue(teststat)
    CLb = b_only_distribution.pvalue(teststat)
    CLs = CLsb / CLb
    CLsb, CLb, CLs = (
        tensorlib.reshape(CLsb, (1,)),
        tensorlib.reshape(CLb, (1,)),
        tensorlib.reshape(CLs, (1,)),
    )

    _returns = [CLs]
    if return_tail_probs:
        _returns.append([CLsb, CLb])
    if return_expected_set:
        CLs_exp = []
        for n_sigma in [2, 1, 0, -1, -2]:

            expected_bonly_teststat = b_only_distribution.expected_value(n_sigma)

            CLs = sig_plus_bkg_distribution.pvalue(
                expected_bonly_teststat
            ) / b_only_distribution.pvalue(expected_bonly_teststat)
            CLs_exp.append(tensorlib.reshape(CLs, (1,)))
        CLs_exp = tensorlib.astensor(CLs_exp)
        if return_expected:
            _returns.append(CLs_exp[2])
        _returns.append(CLs_exp)
    elif return_expected:
        n_sigma = 0
        expected_bonly_teststat = b_only_distribution.expected_value(n_sigma)

        CLs = sig_plus_bkg_distribution.pvalue(
            expected_bonly_teststat
        ) / b_only_distribution.pvalue(expected_bonly_teststat)
        _returns.append(tensorlib.reshape(CLs, (1,)))
    # Enforce a consistent return type of the observed CLs
    return tuple(_returns) if len(_returns) > 1 else _returns[0]
