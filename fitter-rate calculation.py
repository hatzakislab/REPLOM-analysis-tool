"""Script to fit various curves to single aggregate growth data"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import inspect
from iminuit import Minuit
import scipy.stats as stats
import warnings
from scipy.integrate import odeint

warnings.filterwarnings("ignore")


def Sig_parabola(x, G1, G2, L, t0, t1, c0):
    """Computes a growth curve which has the following three regimes
    - Constant growth
    - Growth rate proportional to t
    - Growth rate proportional to t with saturation

    Parameters
    ----------
    x : 1D List-like
        Times at which to obtain a solution
    G1 : Float
        Constant growth rate
    G2 : Float
        Growth rate in second phase
    L : Float
        Saturation point
    t0 : type
        Time for switch from constant growth to growthrate proportional to t
    t1 : Float
        Time for switch from rate proportional to t to rate with saturation
    c0 : Float
        Initial concentration

    Returns
    -------
    1D List-like
        Growth curve

    """

    def rate(x, t):
        if t < t0:
            return np.pi * G1 ** 2
        elif t0 <= t < t1:
            return 0.5 * np.pi * G2 ** 2 * t
        else:
            return (
                0.5
                * np.pi
                * G2 ** 2
                * t
                * (1 / (1 + np.exp((1 / (L / 5)) * (t - L - t1))))
            )

    sol = odeint(rate, c0, np.concatenate([[x[0]], x]))
    return sol[1:][:, 0]


def Sig_two_line(x, G1, G2, L, t0, t1, c0):
    """Computes a growth curve which has the following three regimes
    - Constant growth with rate G1
    - Constant growth with rate G2
    - Constant growth with rate G2 and saturation

    Parameters
    ----------
    x : 1D List-like
        Times at which to obtain a solution
    G1 : Float
        Constant growth rate 1
    G2 : Float
        Constant growth rate 2
    L : Float
        Saturation point
    t0 : type
        Time for switch from growth with rate G1 to G2
    t1 : Float
        Time for switch from growth with rate G2 to G2 with saturation
    c0 : Float
        Initial concentration

    Returns
    -------
    1D List-like
        Growth curve

    """

    def rate(x, t):
        if t < t0:
            return G1
        elif t0 <= t < t1:
            return G2
        else:
            return G2 * (1 / (1 + np.exp((1 / (L / 5)) * (t - L - t1))))

    sol = odeint(rate, c0, np.concatenate([[x[0]], x]))
    return sol[1:][:, 0]


def Sig_straight(x, G, L, t0, c0):
    """Computes a growth curve which has the following two regimes
    - Constant growth with rate G
    - Constant growth with rate G and saturation

    Parameters
    ----------
    x : 1D List-like
        Times at which to obtain a solution
    G : Float
        Constant growth rate 1
    L : Float
        Saturation point
    t0 : Float
        Time for switch from constant growth to growthrate with saturation
    c0 : Float
        Initial concentration

    Returns
    -------
    1D List-like
        Growth curve

    """

    def rate(x, t):
        if t < t0:
            return G
        else:
            return G * (1 / (1 + np.exp((1 / (L / 5)) * (t - L - t0))))

    sol = odeint(rate, c0, np.concatenate([[0], x]))
    return sol[1:][:, 0]


def format_value(value, decimals):
    """
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """

    if isinstance(value, (float, np.float)):
        return f"{value:.{decimals}f}"
    elif isinstance(value, (int, np.integer)):
        return f"{value:d}"
    else:
        return f"{value}"


def values_to_string(values, decimals):
    """
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'.
    """

    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f"{tmp[0]} +/- {tmp[1]}")
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """Returns the length of the longest string in a list of strings"""
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.
    """

    names = d.keys()
    max_names = len_of_longest_string(names)

    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)

    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1
        string += "{name:s} {value:>{spacing}} \n".format(
            name=name, value=value, spacing=spacing
        )
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color="k"):
    """Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(
        x_coord,
        y_coord,
        string,
        family="monospace",
        fontsize=fontsize,
        transform=ax.transAxes,
        verticalalignment="top",
        color=color,
    )
    return None


def Chi2Fit(
    x,
    y,
    sy,
    f,
    plot=True,
    print_level=0,
    labels=None,
    ax=None,
    savefig="",
    valpos=None,
    exponential=False,
    fitcol=None,
    markersize=5,
    name=None,
    fontsize=15,
    linewidth=3,
    **guesses,
):
    """Function that peforms a Chi2Fit to data given function
    ----------
    Parameters
    ----------
    x: ndarray of shape for input in f
        - input values to fit
    y: ndarray of shape output from f
        - output values to fit
    sy: ndarray of length y
        - errors on the y values
    f: function
        - Function to fit, should be of form f(x,args), where args
          is a list of arguments
    **guesses: mappings ie. p0=0.1,p1=0.2
        - initial guesses for the fit parameters
    print_level: int 0,1
        - Wether to print output from chi2 ect.
    labels:
        - Mappable to pass to ax.set call to set labels on plot
    name: str
        -Label to call fit in legend
    fontsize: int
        - Size of font in plot
    linewidth: float
        - Width of line on data
    ---------
    Returns
    ---------
    params: length args
        - fit params
    errs: lenght args
        - errror on fit params
    pval: float
        -pvalue for the fit
    """
    names = inspect.getargspec(f)[0][1:]

    def fcn(*invals):
        return np.sum(((f(x, *invals) - y) / sy) ** 2)

    argstr = ", ".join(names)
    fakefunc = "def func(%s):\n    return real_func(%s)\n" % (argstr, argstr)
    fakefunc_code = compile(fakefunc, "fakesource", "exec")
    fakeglobals = {}
    eval(fakefunc_code, {"real_func": fcn}, fakeglobals)
    f_with_good_sig = fakeglobals["func"]

    fcn.errordef = Minuit.LEAST_SQUARES

    chi2_object = f_with_good_sig  # Chi2Regression(f,x,y,sy)
    if len(guesses) != 0:
        minuit = Minuit(chi2_object, **guesses)
    else:
        minuit = Minuit(chi2_object)
    minuit.migrad()
    chi2 = minuit.fval
    Ndof = len(x) - len(guesses)
    Pval = stats.chi2.sf(chi2, Ndof)
    params = minuit.values
    errs = minuit.errors

    if not exponential:
        dict = {"chi2": chi2, "Ndof": Ndof, "Pval": Pval}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2f} +/- {py:4.2f}"
    else:
        dict = {"chi2": f"{chi2:4.4E}", "Ndof": f"{Ndof:4.4E}", "Pval": f"{Pval:4.4E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.4E} +/- {py:4.4E}"
    if plot:
        # Plot the fit
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        text = nice_string_output(dict)
        if valpos is None:
            if fitcol is None:
                add_text_to_ax(0.05, 0.9, text, ax, fontsize=fontsize)
            else:
                add_text_to_ax(0.05, 0.9, text, ax, color=fitcol, fontsize=fontsize)
        else:
            add_text_to_ax(
                valpos[0], valpos[1], text, ax, color=fitcol, fontsize=fontsize
            )
        xmin, xmax = np.min(x), np.max(x)
        if name is None:
            if fitcol is None:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    fmt=".r",
                    label="Data",
                    capsize=2,
                    markersize=markersize,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    marker=".",
                    color=fitcol,
                    label="Data",
                    capsize=2,
                    markersize=markersize,
                )
        else:
            if fitcol is None:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    fmt=".r",
                    capsize=2,
                    markersize=markersize,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    marker=".",
                    color=fitcol,
                    capsize=2,
                    markersize=markersize,
                )
        x_fit = np.linspace(xmin, xmax, 200)
        y_fit = [f(i, *params) for i in x_fit]
        if labels is None:
            ax.set(xlabel="x", ylabel="f(x)")
        else:
            ax.set(**labels)
        if fitcol is None:
            if name is None:
                ax.plot(x_fit, y_fit, color="r", label="Fit", linewidth=linewidth)
            else:
                ax.plot(x_fit, y_fit, color="r", label=name, linewidth=linewidth)
        else:
            if name is None:
                ax.plot(x_fit, y_fit, color=fitcol, label="Fit", linewidth=linewidth)
            else:
                ax.plot(x_fit, y_fit, color=fitcol, label=name, linewidth=linewidth)
        ax.grid()
        if savefig != "":
            plt.savefig(savefig + ".pdf", dpi=500)
        if ax is None:
            plt.legend()
            plt.show()
    return params, errs, Pval


def twoline(x, x01, r1, Offset, r2, switch):
    """Computes a growth curve which has the following two regimes
    - Constant growth with rate r1
    - Constant growth with rate r2

    Parameters
    ----------
    x : 1D List-like
        Times at which to obtain a solution
    x01 : Float
        Initial time from which to compute trace
    r1 : Float
        Growth rate 1
    Offset : float
        Vertical offset of curve
    r2 : Float
        Growth rate 2
    switch : Float
        Time for switch from rate r1 to r2

    Returns
    -------
    1D List-like
        Growth curve

    """
    x02 = switch - (switch - x01) * r1 / r2
    if x < switch:
        return (x - x01) * r1 + Offset
    else:
        return (x - x02) * r2 + Offset


def line(x, x0, r1, Offset):
    return (x - x0) * r1 + Offset


def exponential(x, tau, C, Offset, x0):
    return C * np.exp(tau * (x - x0)) + Offset


def Menten(x, Vmax, Km, Offset):
    return (Vmax * (x - 0)) / (Km + (x - 0)) + Offset


def Fit_G(
    x, y, sy, start_frame=0, t1=100, G1=200, G2=200, L=70, t0=20, func=Sig_parabola
):
    """Fits a function to an observed time series

    Parameters
    ----------
    x : list-like
        Time of observation
    y : list-like
        Observed concentrations
    sy : list-like
        Error bars on observations
    start_frame : int, default=0
        Which index to start fitting from
    t1 : float
        See fit curves for more info
    G1 : float
        See fit curves for more info
    G2 : float
        See fit curves for more info
    L : float
        See fit curves for more info
    t0 : float
        See fit curves for more info
    func : callable
        Function to fit

    Returns
    -------
     param_iso : iterable
        Fit parameters
    errs_iso : iterable
        Errors on fit parameters

    """
    x_fit, y_fit, sy_fit = x[start_frame:], y[start_frame:], sy[start_frame:]
    if func == Sig_parabola or func == Sig_two_line:

        def Chi2_iso(G1, G2, L, t0, t1):
            return np.sum(
                ((func(x_fit - x_fit[0], G1, G2, L, t0, t1, y[0]) - y_fit) / sy_fit)
                ** 2
            )

        Chi2_iso.errordef = Minuit.LEAST_SQUARES

        m = Minuit(Chi2_iso, G1=G1, G2=G2, L=L, t0=t0, t1=t1)
        m.limits["t0"] = (0, x[-1])
        m.limits["G1"] = (0, None)
        m.limits["G2"] = (0, None)
        m.limits["t1"] = (0, x[-1])
        m.limits["L"] = (2, x[-1] / 2)
        m.migrad()  # run optimiser
    else:

        def Chi2_iso(G, L, t0):
            return np.sum(
                ((func(x_fit - x_fit[0], G, L, t0, y[0]) - y_fit) / sy_fit) ** 2
            )

        Chi2_iso.errordef = Minuit.LEAST_SQUARES

        m = Minuit(Chi2_iso, G=G2, L=L, t0=t0)
        m.limits["t0"] = (0, x[-1])
        m.limits["G"] = (0, None)
        m.limits["L"] = (2, None)
        m.migrad()  # run optimiser

    param_iso, errs_iso = m.values, m.errors
    return param_iso, errs_iso


def errfitter(x, y, type="twoline", plotit=False, startval=0, savefig=None, **guesses):
    """Fits a function to data in a more automated fashion than Fit_G

    Parameters
    ----------
    x : list-like
        Time of observation
    y : list-like
        Observed concentrations
    type : str, default=twoline
        Fit type choices are (if not a known function it defaults to straightline fit)
            decaying_line :
                Fits the Sig_straight function
            decaying_parabola :
                Fits the Sig_parabola function
            decaying_twoline :
                Fits the Sig_two_line function
            twoline :
                Fits the twoline function
            exp :
                Fits the exponential function
            Menten :
                Fits the Menten function
            if not one of the above :
                Fits the line function
    plotit : boolean
        Wether to display a plot of the fit
    startval : integer
        index to start the fit
    savefig : str
        path to save the figure to
    **guesses : mappable
        initial guesses for parameter values in the corresponding fit function

    Returns
    -------
    iterable
        [params, errs, pval]

    """

    if type == "decaying_line":
        x_fit = x - x[0]
        params, errs = Fit_G(
            x_fit, y, np.ones(x_fit.shape), func=Sig_straight, **guesses
        )
        params = list(params) + [y[0]]
        sy = np.ones(len(x_fit)) * np.percentile(y - Sig_straight(x_fit, *params), 50)

        params, errs = Fit_G(x_fit, y, sy, func=Sig_straight, **guesses)
        params = list(params) + [y[0]]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        # ax.errorbar(x, y, sy, linewidth=0.1, elinewidth=0.1)
        ax.plot(x, y, "o", linewidth=4, c="k")
        ax.plot(x, Sig_straight(x_fit, *params), linewidth=2, c="red")
        ax.set(xlabel="time / s", ylabel=r"Area [$\mathrm{nm}^2/s$]")
        # ax.axvline(x=params[-1]+x[0],c="k",linestyle="--")
        # ax.axvline(x=params[-1]+params[-2]+x[0], c="darkblue", linestyle="--")

        params[-1] += x[0]
        dict = {}
        names = ["G", "L", "t0"]
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2E} +/- {py:4.2E}"
        text = nice_string_output(dict, extra_spacing=0)
        add_text_to_ax(0.01, 0.99, text, ax, fontsize=10)
        fig.savefig(savefig + ".pdf")
        return [params, errs, None]
    elif type == "decaying_parabola":
        x_fit = x - x[0]
        params, errs = Fit_G(
            x_fit, y, np.ones(x_fit.shape), func=Sig_parabola, **guesses
        )
        params = list(params) + [y[0]]
        sy = np.ones(len(x_fit)) * np.percentile(y - Sig_parabola(x_fit, *params), 50)
        params, errs = Fit_G(x_fit, y, sy, func=Sig_parabola, **guesses)
        params = list(params) + [y[0]]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(x, y, "o", linewidth=4, c="k")
        ax.plot(x, Sig_parabola(x_fit, *params), linewidth=2, c="red")
        ax.set(xlabel="time / s", ylabel=r"Area [$\mathrm{nm}^2/s$]")
        ax.axvline(x=params[-2] + x[0], c="k", linestyle="--")
        ax.axvline(x=params[-1] + x[0], c="k", linestyle="--")
        ax.axvline(x=params[-1] + params[-3] + x[0], c="darkblue", linestyle="--")

        params[-1] += x[0]
        params[-2] += x[0]
        dict = {}
        names = ["G1", "G2", "L", "t0", "t1"]
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2E} +/- {py:4.2E}"
        text = nice_string_output(dict, extra_spacing=0)
        add_text_to_ax(0.01, 0.99, text, ax, fontsize=10)
        fig.savefig(savefig + ".pdf")
        return [params, errs, None]
    elif type == "decaying_twoline":
        x_fit = x - x[0]
        params, errs = Fit_G(
            x_fit, y, np.ones(x_fit.shape), func=Sig_two_line, **guesses
        )
        params = list(params) + [y[0]]
        sy = np.ones(len(x_fit)) * np.percentile(y - Sig_two_line(x_fit, *params), 50)
        params, errs = Fit_G(x_fit, y, sy, func=Sig_two_line, **guesses)
        params = list(params) + [y[0]]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.errorbar(x, y, sy, linewidth=3, elinewidth=1)
        ax.plot(x, Sig_two_line(x_fit, *params), linewidth=3, c="red")
        ax.set(xlabel="time / s", ylabel=r"Area [$\mathrm{nm}^2/s$]")
        ax.axvline(x=params[-3] + x[0], c="k", linestyle="--")
        ax.axvline(x=params[-2] + x[0], c="k", linestyle="--")
        ax.axvline(x=params[-2] + params[-4] + x[0], c="darkblue", linestyle="--")

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(x, y, "o", linewidth=4, c="k")
        ax.plot(x, Sig_two_line(x_fit, *params), linewidth=2, c="red")
        ax.set(xlabel="time / s", ylabel=r"Area [$\mathrm{nm}^2/s$]")
        ax.axvline(x=params[-3] + x[0], c="k", linestyle="--")
        ax.axvline(x=params[-2] + x[0], c="k", linestyle="--")
        ax.axvline(x=params[-2] + params[-4] + x[0], c="darkblue", linestyle="--")

        params[-1] += x[0]
        params[-2] += x[0]
        dict = {}
        names = ["G1", "G2", "L", "t0", "t1"]
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2E} +/- {py:4.2E}"
        text = nice_string_output(dict, extra_spacing=0)
        add_text_to_ax(0.01, 0.99, text, ax, fontsize=10)
        fig.savefig(savefig + ".pdf")
        return [params, errs, None]

    elif type == "twoline":
        params, errs, pval = Chi2Fit(
            x, y, np.ones(x.shape), twoline, plot=False, **guesses
        )
        sy = np.percentile(y - np.array([twoline(i, *params) for i in x]), 50)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        params, errs, pval = Chi2Fit(
            x, y, sy * np.ones(x.shape), twoline, ax=ax, exponential=True, **guesses
        )
        ax.grid()
        fig.savefig(savefig + ".pdf")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        xmin, xmax = np.min(x), np.max(x)
        x_fit = np.linspace(xmin, xmax, 200)
        y_fit = [twoline(i, *params) for i in x_fit]
        ax.errorbar(
            x,
            y,
            yerr=sy,
            linestyle="",
            ecolor="k",
            fmt=".r",
            label="Data",
            capsize=2,
            markersize=5,
        )
        ax.plot(x_fit, y_fit, color="r", label="Fit", linewidth=3)
        ax.set(xlabel="time/s")
        ax.set(ylabel="area/nm^2")
        fig.savefig(savefig + "_modified_.pdf")
        return [params, errs, None]
    elif type == "exp":
        params, errs, pval = Chi2Fit(
            x, y, np.ones(x.shape), exponential, plot=False, **guesses
        )
        sy = np.percentile(y - np.array([exponential(i, *params) for i in x]), 50)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        params, errs, pval = Chi2Fit(
            x, y, sy * np.ones(x.shape), exponential, ax=ax, exponential=True, **guesses
        )
        ax.grid()
        fig.savefig(savefig + ".pdf")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        xmin, xmax = np.min(x), np.max(x)
        x_fit = np.linspace(xmin, xmax, 200)
        y_fit = [exponential(i, *params) for i in x_fit]
        ax.errorbar(
            x,
            y,
            yerr=sy,
            linestyle="",
            ecolor="k",
            fmt=".r",
            label="Data",
            capsize=2,
            markersize=5,
        )
        ax.plot(x_fit, y_fit, color="r", label="Fit", linewidth=3)
        ax.set(xlabel="time/s")
        ax.set(ylabel="area/nm^2")
        fig.savefig(savefig + "_modified_.pdf")

        return [params, errs, pval]
    elif type == "Menten":
        params, errs, pval = Chi2Fit(
            x, y, np.ones(x.shape), Menten, plot=False, **guesses
        )
        sy = np.percentile(y - np.array([Menten(i, *params) for i in x]), 50)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        params, errs, pval = Chi2Fit(
            x, y, sy * np.ones(x.shape), Menten, ax=ax, exponential=True, **guesses
        )
        ax.grid()
        fig.savefig(savefig + ".pdf")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        xmin, xmax = np.min(x), np.max(x)
        x_fit = np.linspace(xmin, xmax, 200)
        y_fit = [Menten(i, *params) for i in x_fit]
        ax.errorbar(
            x,
            y,
            yerr=sy,
            linestyle="",
            ecolor="k",
            fmt=".r",
            label="Data",
            capsize=2,
            markersize=5,
        )
        ax.plot(x_fit, y_fit, color="r", label="Fit", linewidth=3)
        ax.set(xlabel="time/s")
        ax.set(ylabel="area/nm^2")
        fig.savefig(savefig + "_modified_.pdf")

        return [params, errs, pval]
    else:
        params, errs, pval = Chi2Fit(
            x, y, np.ones(x.shape), line, plot=False, **guesses
        )
        sy = np.percentile(y - np.array([line(i, *params) for i in x]), 50)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        params, errs, pval = Chi2Fit(
            x, y, sy * np.ones(x.shape), line, ax=ax, exponential=True, **guesses
        )
        ax.grid()
        fig.savefig(savefig + ".pdf")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        xmin, xmax = np.min(x), np.max(x)
        x_fit = np.linspace(xmin, xmax, 200)
        y_fit = [line(i, *params) for i in x_fit]
        ax.errorbar(
            x,
            y,
            yerr=sy,
            linestyle="",
            ecolor="k",
            fmt=".r",
            label="Data",
            capsize=2,
            markersize=5,
        )
        ax.plot(x_fit, y_fit, color="r", label="Fit", linewidth=3)
        ax.set(xlabel="time/s")
        ax.set(ylabel="Length/nm^2")
        fig.savefig(savefig + "_modified_.pdf")
        return [params, errs, pval]


import os

"""Fit an anisotropic aggregate"""
dat = np.genfromtxt("example_raw data/Group 1 Growth curve")
startval = 0
dat = dat[(dat[:, 0] > startval)]
# dat = dat[(dat[:, 0] < 398)]

p, errs, pval = errfitter(
    dat[:, 0],
    dat[:, 1],
    plotit=True,
    startval=startval,
    type="decaying_twoline",
    savefig="Fit_anisotropic",
    **{"G1": 2e4, "G2": 1.7e5, "L": 50, "t0": 150, "t1": 300},
)

"""Fit an isotropic aggregate"""
dat = np.genfromtxt("example_raw data/Group 0 Growth curve")
startval = 60
dat = dat[(dat[:, 0] > startval)]

p, errs, pval = errfitter(
    dat[:, 0],
    dat[:, 1],
    plotit=True,
    startval=startval,
    type="decaying_line",
    savefig="Fit_anisotropic",
    **{"G2": 7e5, "L": 50, "t0": 150},
)
