#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 15:07:43

import sys
import shutil
from pathlib import Path

def str2bool(flag):
    if flag.lower() == 'true':
        return True
    elif flag.lower() == 'false':
        return False
    else:
        sys.exit('Please Input correct flag: True or False!')

def str2none(flag):
    if flag.lower() == 'none':
        return None
    else:
        return flag

def update_args(args_json, args_parser):
    for arg in vars(args_parser):
        args_json[arg] = getattr(args_parser, arg)

