##########################################################################################
# Description: Utility functions for statistical data analysis.
##########################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
from scipy.stats import chi2_contingency, shapiro, norm, iqr
from typing import Union, List, Tuple, Dict

__all__ = ["summarize_iccs", "summarize_average", "chi2_test", "bland_altman_plot"]

################################################################################
# -F: summarize_iccs


def summarize_iccs(
    df_dict: Dict[str, pd.DataFrame],
    targets: str,
    ratings: Union[str, List[str]],
    print_results: bool = False,
    return_str_result: bool = True,
    icc_type: str = "ICC2",
    nan_policy: str = "omit",
    bracket: str = "parentheses",
):
    """
    Author: Linjun Yang. Ph.D.
    
    This is a custom Python function based on pingouin.intraclass_corr to calculate ICC values among multiple raters.
    The function calculates ICC values for each rating/measurement in the ratings list. User needs to prepare the dataframes\
    for each rater and provide them in the df_dict dictionary. Each dataframe should have the column for each rating/measurement\
    and the target variable (which specify the samples/cases on which the measurements were performed). User can specify the type\
    of ICC values to calculate. The function return the calculated ICCs and 95% confidence intervals, as well as the string version\
    of the result, e.g., '0.91 (0.8, 0.94)' for table creation.

    Args:
        df_dict (dict[str, pd.DataFrame]): dictionary of dataframes, e.g., {"observer 1": df1, "observer 2": df2, "AI": df_AI}
        targets (str): argument taken by pg.intraclass_corr; the column name of the target variable
        ratings (Union[str, list[str]]): list of ratings/measurement to calculate ICC values
        print_results (bool, optional): if to print results. Defaults to False.
        return_str_result (bool, optional): if to generate string of resulting ICC values for table generation. Defaults to True.
        icc_type (str, optional): type of ICC value to calculate. Defaults to "ICC2" (Single random raters).
        nan_policy (str, optional): how to handle NaN value when using pg.intraclass_corr. Defaults to "omit".
        bracket (str, optional): type of bracket to use for the confidence interval. Defaults to "parentheses".

    Returns:
        dict: result_dict of rating: (ICC, CI95) and str_result_dict of rating: "ICC [CI95]"
    """

    # type of ICC value to use
    icc_type_dict = {"ICC1": 0, "ICC2": 1, "ICC3": 2, "ICC1k": 3, "ICC2k": 4, "ICC3k": 5}
    icc_id = icc_type_dict[icc_type]

    # in case that only one rating/measurement for which the ICC is calculated
    if isinstance(ratings, str):
        ratings = [ratings]

    # list of raters, e.g., ["observer 1", "observer 2", "AI"]; the keys of the df_dict
    rater_list = list(df_dict.keys())

    # concatenate the dataframes; create a copy to avoid touching the original dataframes
    df_sequence = tuple(df[[targets] + ratings].copy() for df in df_dict.values())
    df_all = pd.concat(df_sequence, axis=0)

    # assign raters to the concatenated dataframe
    raters = list()
    for i, df_current in enumerate(df_sequence):
        rater = rater_list[i]
        raters += [rater for _ in range(len(df_current))]
    df_all["rater"] = raters

    # Go through the ratings/measurements
    result_dict = dict()
    for rating in ratings:
        icc_result = pg.intraclass_corr(
            data=df_all,
            targets=targets,
            raters="rater",
            ratings=rating,
            nan_policy=nan_policy,
        )

        # post-process ICC and its 95% confidence interval based on selected icc id
        icc = np.clip(icc_result.loc[icc_id, "ICC"], 0.0, 1.0)
        ci95 = np.clip(icc_result.loc[icc_id, "CI95%"], 0.0, 1.0)
        result_dict[rating] = (icc, ci95)

    # print the results if print_results is True
    if print_results:
        # e.g., "Observer 1, Observer 2, and AI"
        raters_string = ""
        num_raters = len(rater_list)
        for i, rater in enumerate(rater_list):
            if i < len(rater_list) - 1:
                raters_string += rater + ", " if num_raters > 2 else rater + " "
            else:
                raters_string += f"and {rater}"

        preposition = "between" if num_raters == 2 else "among"
        for rating in ratings:
            print(
                f"ICC for {rating} {preposition} {raters_string}: {result_dict[rating][0]:.2f} ({result_dict[rating][1]})"
            )

    # prepare the string of results to output table directly
    if return_str_result:
        str_result_dict = dict()
        for rating in ratings:
            icc_str = f"{result_dict[rating][0]:.2f}"
            ci95_str = " ".join([str(np.round(val, 2)) for val in result_dict[rating][1]])
            if bracket == "parentheses":
                str_result_dict[rating] = f"{icc_str} ({ci95_str})"
            else:
                str_result_dict[rating] = f"{icc_str} [{ci95_str}]"
        return result_dict, str_result_dict
    else:
        return result_dict


def summarize_average(
    data: np.ndarray, 
    return_ci: bool = False
) -> Dict[str, Union[float, Tuple[float, float], str]]:
    """
    Author: Linjun Yang. Ph.D.
    
    Summarizes a 1D NumPy array by checking its normality and calculating
    mean & std or median & IQR accordingly. Optionally returns 95% CI.
    
    Parameters:
        data (np.ndarray): 1D array of numerical data.
        return_ci (bool): Whether to calculate 95% confidence interval.
    
    Returns:
        dict: Summary containing either mean/std or median/IQR,
              and optionally the confidence interval.
    """
    if data.ndim != 1:
        raise ValueError("Input array must be 1D.")

    # Normality test
    stat, p_value = shapiro(data)
    normal = p_value > 0.05

    summary = {}

    if normal:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        summary["distribution"] = "normal"
        summary["mean"] = mean
        summary["std"] = std
        if return_ci:
            se = std / np.sqrt(len(data))
            ci = norm.interval(0.95, loc=mean, scale=se)
            summary["95% CI"] = ci
    else:
        median = np.median(data)
        iqr_val = iqr(data)
        summary["distribution"] = "non-normal"
        summary["median"] = median
        summary["IQR"] = iqr_val
        if return_ci:
            # Bootstrapped CI for the median
            boot_medians = np.array([
                np.median(np.random.choice(data, size=len(data), replace=True))
                for _ in range(1000)
            ])
            ci = np.percentile(boot_medians, [2.5, 97.5])
            summary["95% CI"] = tuple(ci)

    return summary


def chi2_test(
    df: pd.DataFrame, group_col: str, class_col: str, print_results: bool = False
):
    """
    Perform a chi-square test of independence on a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - group_col (str): The name of the column representing the grouping variable.
    - class_col (str): The name of the column representing the class variable.
    - print_results (bool): Whether to print the test results. Default is False.

    Returns:
    - chi2 (float): The chi-square test statistic.
    - p (float): The p-value of the test.
    - dof (int): The degrees of freedom.
    - expected (ndarray): The expected frequencies table.

    Example:
    >>> df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'], 'Class': ['X', 'Y', 'X', 'Y']})
    >>> chi2_test(df, 'Group', 'Class', print_results=True)
    Chi2 Statistic: 0.5
    P-value: 0.479
    Degrees of Freedom: 1
    Expected Frequencies Table:
    [[0.5 0.5]
     [0.5 0.5]]
    (0.5, 0.479, 1, array([[0.5, 0.5],
           [0.5, 0.5]]))
    """

    # Create a contingency table
    contingency_table = pd.crosstab(df[group_col], df[class_col])

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    if print_results:
        print(f"Chi2 Statistic: {chi2}")
        print(f"P-value: {p}")
        print(f"Degrees of Freedom: {dof}")
        print("Expected Frequencies Table:")
        print(expected)

    return chi2, p, dof, expected


def bland_altman_plot(data1, data2, title, *args, **kwargs):
    """
    Author: Kellen Mulford Ph.D.

    Args:
        data1: data1
        data2: data2
        title: title of the figure

    Returns:
        matplotlib.figure.Figure: whole figure of bland altman plot
    """
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    fig = plt.figure()
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.title(title)
    
    return fig
