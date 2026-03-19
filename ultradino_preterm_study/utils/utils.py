"""Utility functions for the GROWTH project"""

# pylint: disable=import-error
import numpy as np  # type: ignore
import scipy.ndimage  # type: ignore
import scipy.stats  # type: ignore
from torch.utils.tensorboard import SummaryWriter  # type: ignore

# pylint: disable=invalid-name
# pylint: disable=too-many-locals


def dict2device(d, device):
    """Move all tensors in a dictionary to a device"""
    for key in d.keys():
        d[key] = d[key].to(device)


def initWriter(args):
    """Initialize a tensorboard writer"""
    report = "# Input arguments  \n"
    report_general = "## General \n"
    report_model = "## Model \n"
    report_training = "## Training \n"
    report_optimizer = "## Optimizer \n"
    report_loss = "## Loss function \n"
    args_dict = vars(args)

    # andrea: maybe easier with a dict?
    for key in args_dict.keys():
        if key[:2] == "g1":
            report_general += f"- {key[3:]}: {args_dict[key]}  \n"
        elif key[:2] == "g2":
            report_model += f"- {key[3:]}: {args_dict[key]}  \n"
        elif key[:2] == "g3":
            report_training += f"- {key[3:]}: {args_dict[key]}  \n"
        elif key[:2] == "g4":
            report_optimizer += f"- {key[3:]}: {args_dict[key]}  \n"
        elif key[:2] == "g5":
            report_loss += f"- {key[3:]}: {args_dict[key]}  \n"
        else:
            raise ValueError(f"Unexpected parameter group {key[:2]}")

    if args.g1_experiment_name != "":  # andrea: could be simplified to bool
        writer = SummaryWriter("runs/" + args.g1_experiment_name)
    else:
        writer = SummaryWriter()

    report += report_general
    report += report_model
    report += report_training
    report += report_optimizer
    report += report_loss

    writer.add_text("Arguments", report)

    return writer


# andrea: _getHadlock_BPD_HC_AC_FL, _getHadlock_HC_AC_FL and getHadlock
# can be merged into a single function.
def _getHadlock_BPD_HC_AC_FL(bpd, hc, ac, fl):
    """Hadlock formula for weight estimation"""

    # The formula uses values in cm as opposed to mm. Convert
    hc, bpd, ac, fl = hc / 10, bpd / 10, ac / 10, fl / 10

    log10weight = (
        1.3596
        - 0.00386 * ac * fl
        + 0.0064 * hc
        + 0.00061 * bpd * ac
        + 0.0424 * ac
        + 0.174 * fl
    )

    try:
        weight = 10**log10weight
    except OverflowError:
        weight = np.nan

    return weight


def _getHadlock_HC_AC_FL(hc, ac, fl):
    """Hadlock formula for weight estimation"""
    # The formula uses values in cm as opposed to mm. Convert
    hc, ac, fl = hc / 10, ac / 10, fl / 10

    log10weight = 1.326 - 0.00326 * ac * fl + 0.0107 * hc + 0.0438 * ac + 0.158 * fl

    try:
        weight = 10**log10weight
    except OverflowError:
        weight = np.nan

    return weight


def getHadlock(hc, bpd, ac, fl, mode):
    """

    Parameters
    ----------
    Acceptable input parameter types: int, float, np.array, torch.tensor
    It is assumed that the dimensions match.

    hc   : 'Head Circumference [mm]'
    bpd  : 'Biparietal DIameter [mm]'
    ac   : 'Abdominal Circumference [mm]'
    fl   : 'Femur Length [mm]'
    mode : "'hc ac fl' or 'bpd hc ac fl' [str]"


    Returns
    -------
    weight : 'Hadlock Weight Estimate [g]', Matches input type

    """
    if mode == "bpd_hc_ac_fl":
        return _getHadlock_BPD_HC_AC_FL(bpd, hc, ac, fl)
    if mode == "hc_ac_fl":
        return _getHadlock_HC_AC_FL(hc, ac, fl)
    raise ValueError(
        f"Mode: {mode} is not supported. Input 'hc_ac_fl' or 'bpd_hc_ac_fl'! "
    )


# andrea: girlGrowthCurve, boyGrowthCurve could be merged into a function with attrs "boy", "girl".
def girlGrowthCurve(ga):
    """
    Parameters
    ----------
    ga : int, np.array, torch.tensor
        Gestational age [days]

    Returns
    -------
    Matches input type.
        Mean weight [grams] at a given gestational age.
    """
    return (
        -2.761948 * 10 ** (-6) * ga**4
        + 1.744841 * 10 ** (-3) * ga**3
        - 2.893626 * 10 ** (-1) * ga**2
        + 1.891197 * 10 ** (1) * ga
        - 4.135122 * 10 ** (2)
    )


def boyGrowthCurve(ga):
    """
    Parameters
    ----------
    ga : int, np.array, torch.tensor
        Gestational age [days]

    Returns
    -------
    Matches input type.
        Mean weight [grams] at a given gestational age.
    """
    return (
        -1.907345 * 10 ** (-6) * ga**4
        + 1.140644 * 10 ** (-3) * ga**3
        - 1.336265 * 10 ** (-1) * ga**2
        + 1.976961 * 10 ** (0) * ga
        + 2.410054 * 10 ** (2)
    )


# The rest of the script is only used in the worker file and not in the actual code
def cohen_d(x1, x2):
    """
    Calculate Cohen's d.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.

    Returns
    -------
    d : float
        Cohen's d.
    """
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)  # sample variance
    s = np.sqrt(
        ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    )  # pooled standard deviation
    u1, u2 = np.mean(x1), np.mean(x2)
    return (u1 - u2) / s


def printComparison(scan_weight, model, hadlock, model_hadlock):
    """Prints the comparison between the model and the hadlock formula"""
    _sw = scan_weight
    _swm = model
    _swh = hadlock
    _swmh = model_hadlock

    _swmh_comb = (_swm + _swmh) / 2
    # _remh_comb = abs(_sw-_swmh_comb)/_sw * 100
    _remh = abs(_sw - _swmh) / _sw * 100
    _rem = abs(_sw - _swm) / _sw * 100
    _reh = abs(_sw - _swh) / _sw * 100

    N1 = _sw.shape[0]

    abs_model = abs(_sw - _swm)
    abs_hadlock = abs(_sw - _swh)
    Model_better = np.sum(abs_model < abs_hadlock)
    proportion = Model_better / len(_sw)

    print(f"Number of samples {N1}")

    print(
        f"\nProportion of samples where model does better than hadlock: {proportion}\n"
    )

    print("Hadlock")
    print(f"Mean absolute error: {abs(_sw-_swh).mean()} gram")
    print(f"Mean relative error: {_reh.mean()} %")
    print(f"STD relative error: {_reh.std()} %")
    print("-" * 50)

    print("Model")
    print(f"Mean absolute error: {abs(_sw-_swm).mean()} gram")
    print(f"Mean relative error: {_rem.mean()} %")
    print(f"STD relative error: {_rem.std()} %")
    _stat, pval = scipy.stats.ttest_ind(_reh, _rem, equal_var=False)
    print(f"p-val {pval}")
    print(f"Cohen's d: {cohen_d(_reh,_rem)}")
    print("-" * 50)

    print("Model Hadlock")
    print(f"Mean absolute error: {abs(_sw-_swmh).mean()} gram")
    print(f"Mean relative error: {_remh.mean()} %")
    print(f"STD relative error: {_remh.std()} %")
    _stat, pval = scipy.stats.ttest_ind(_reh, _remh, equal_var=False)
    print(f"p-val {pval}")
    print(f"Cohen's d: {cohen_d(_reh,_remh)}")
    print("-" * 50)
