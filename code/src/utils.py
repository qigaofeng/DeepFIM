# -*- coding:utf-8 -*-

import collections
import json
import logging
from threading import Thread

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


VarLenFeat = collections.namedtuple(
    'VarLenFeat', ['name', 'dimension', 'maxlen', 'combiner'])
SingleFeat = collections.namedtuple(
    'SingleFeat', ['name', 'dimension', ])

def check_feature_config_dict(feature_dim_dict):
    if not isinstance(feature_dim_dict, dict):
        raise ValueError(
            "feature_dim_dict must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}")
    if "sparse" not in feature_dim_dict:
        feature_dim_dict['sparse'] = []
    if "dense" not in feature_dim_dict:
        feature_dim_dict['dense'] = []
    if not isinstance(feature_dim_dict["sparse"], list):
        raise ValueError("feature_dim_dict['sparse'] must be a list,cur is", type(
            feature_dim_dict['sparse']))

    if not isinstance(feature_dim_dict["dense"], list):
        raise ValueError("feature_dim_dict['dense'] must be a list,cur is", type(
            feature_dim_dict['dense']))
