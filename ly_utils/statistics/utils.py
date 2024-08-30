##########################################################################################
# Description: Utility functions for statistical data analysis.
##########################################################################################

import numpy as np
import pandas as pd
import pingouin as pg
from typing import Union, List, Dict

__all__ = ["summarize_iccs"]

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
            str_result_dict[rating] = f"{icc_str} [{ci95_str}]"
        return result_dict, str_result_dict
    else:
        return result_dict
