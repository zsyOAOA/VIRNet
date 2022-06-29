#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-02-06 10:34:59

import shutil
import logging
from pathlib import Path

def make_log(file_path,
             log_name=None,
             formatter='%(message)s',
             file_level=logging.INFO,
             stream_level=None):
    '''
    Input:
        file_path: path of logging file
        log_name: name of logging, if None, root logger, see the help of logging.getLogger
        formatter: logging Formatter
        file_level: logging level for the log file
        stream_level: logging level for printing on the console, If None, disabling printing
    '''
    logger = logging.getLogger(name=log_name)
    logger.setLevel(logging.INFO)   # necessary

    # build the file handler
    fh = logging.FileHandler(file_path, mode='w')
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(formatter))
    # add file handler to logger
    logger.addHandler(fh)

    # printing on the console
    if not stream_level is None:
        ch = logging.StreamHandler()
        ch.setLevel(stream_level)
        ch.setFormatter(logging.Formatter(formatter))
        logger.addHandler(ch)

    return logger

def mkdir(dir_path, delete=False, parents=True):
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    if delete:
        if dir_path.exists():
            shutil.rmtree(str(dir_path))
    if not dir_path.exists():
        dir_path.mkdir(parents=parents)
