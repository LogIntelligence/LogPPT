"""
Description : This file implements the function to evaluation accuracy of log parsing
Author      : LogPAI team
License     : MIT
"""

import sys
import pandas as pd
from collections import defaultdict
try:
    from scipy.misc import comb
except ImportError as e:
    from scipy.special import comb
from tqdm import tqdm


def evaluate(df_groundtruth, df_parsedlog, filter_templates=None):
    """ Evaluation function to org_benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """ 
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    GA, FGA = get_accuracy(df_groundtruth['EventTemplate'], df_parsedlog['EventTemplate'])
    print('Grouping_Accuracy (GA): %.4f, FGA: %.4f,'%(GA, FGA))
    return GA, FGA

def get_accuracy(series_groundtruth, series_parsedlog, filter_templates=None):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('groundtruth')
    accurate_events = 0 # determine how many lines are correctly parsed
    accurate_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    # for ground_truthId in tqdm(series_groundtruth_valuecounts.index):
    for ground_truthId, group in tqdm(grouped_df):
        # logIds = series_groundtruth[series_groundtruth == ground_truthId].index
        series_parsedlog_logId_valuecounts = group['parsedlog'].value_counts()
        if filter_templates is not None and ground_truthId in filter_templates:
            for parsed_eventId in series_parsedlog_logId_valuecounts.index:
                filter_identify_templates.add(parsed_eventId)
        if series_parsedlog_logId_valuecounts.size == 1:
            parsed_eventId = series_parsedlog_logId_valuecounts.index[0]
            if len(group) == series_parsedlog[series_parsedlog == parsed_eventId].size:
                if (filter_templates is None) or (ground_truthId in filter_templates):
                    accurate_events += len(group)
                    accurate_templates += 1
    # print("filter templates: ", len(filter_templates))
    # print("total messages: ", len(series_groundtruth[series_groundtruth.isin(filter_templates)]))
    # print("set of total: ", len(filter_identify_templates))
    # print(accurate_events, accurate_templates)
    if filter_templates is not None:
        GA = float(accurate_events) / len(series_groundtruth[series_groundtruth.isin(filter_templates)])
        PGA = float(accurate_templates) / len(filter_identify_templates)
        RGA = float(accurate_templates) / len(filter_templates)
    else:
        GA = float(accurate_events) / len(series_groundtruth)
        PGA = float(accurate_templates) / len(series_parsedlog_valuecounts)
        RGA = float(accurate_templates) / len(series_groundtruth_valuecounts)
    # print(FGA, RGA)
    FGA = 0.0
    if PGA != 0 or RGA != 0:
        FGA = 2 * (PGA * RGA) / (PGA + RGA)
    return GA, FGA


# def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
#     """ Compute accuracy metrics between log parsing results and ground truth
    
#     Arguments
#     ---------
#         series_groundtruth : pandas.Series
#             A sequence of groundtruth event Ids
#         series_parsedlog : pandas.Series
#             A sequence of parsed event Ids
#         debug : bool, default False
#             print error log messages when set to True

#     Returns
#     -------
#         precision : float
#         recall : float
#         f_measure : float
#         accuracy : float
#     """
#     series_groundtruth_valuecounts = series_groundtruth.value_counts()
#     real_pairs = 0
#     for count in series_groundtruth_valuecounts:
#         if count > 1:
#             real_pairs += comb(count, 2)

#     series_parsedlog_valuecounts = series_parsedlog.value_counts()
#     parsed_pairs = 0
#     for count in series_parsedlog_valuecounts:
#         if count > 1:
#             parsed_pairs += comb(count, 2)

#     accurate_pairs = 0
#     accurate_events = 0 # determine how many lines are correctly parsed
#     accurate_templates = 0
#     for parsed_eventId in series_parsedlog_valuecounts.index:
#         crr_accuarte_events = 0
#         logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
#         series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
#         error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
#         error = True
#         if series_groundtruth_logId_valuecounts.size == 1:
#             groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
#             if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
#                 crr_accuarte_events += logIds.size
#                 error = False
#         accurate_events += crr_accuarte_events
#         if crr_accuarte_events == series_parsedlog[series_parsedlog == parsed_eventId].size:
#             accurate_templates += 1
#         if error and debug:
#             print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
#         for count in series_groundtruth_logId_valuecounts:
#             if count > 1:
#                 accurate_pairs += comb(count, 2)

#     # precision = float(accurate_pairs) / parsed_pairs
#     # recall = float(accurate_pairs) / real_pairs
#     # f_measure = 2 * precision * recall / (precision + recall)
#     GA = float(accurate_events) / series_groundtruth.size
#     PGA = float(accurate_templates) / series_parsedlog_valuecounts.size
#     RGA = float(accurate_templates) / series_groundtruth_valuecounts.size
#     FGA = 2 * (PGA * RGA) / (PGA + RGA)
#     return GA, FGA