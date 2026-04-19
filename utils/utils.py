#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:30:20 2026

@author: jacob
"""

import pandas as pd

def unpack_dict_to_list(dict, dict_key):
    # This function unpacks the list under dict_key
    # and returns a list of entries instead of a dict
    result = []
    first_iter = True
    for key, subdict in dict.items():
        for item in subdict.get(dict_key):
            if first_iter:
                headers = [k for k in subdict.keys() if k != dict_key] + list(item.keys())
                result.append(headers)
                first_iter = False
            new_list = [v for k, v in subdict.items() if k != dict_key] + list(item.values())
            result.append(new_list)
    return result

def unpack_dict_to_DF(dict, dict_key):
    temp_list = unpack_dict_to_list(dict, dict_key)
    df = pd.DataFrame(temp_list[1:], columns=temp_list[0])
    return df

