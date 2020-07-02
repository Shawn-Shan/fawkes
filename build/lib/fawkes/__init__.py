# -*- coding: utf-8 -*-
# @Date    : 2020-07-01
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/


__version__ = '0.0.2'

from .differentiator import FawkesMaskGeneration
from .utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked, \
    Faces
from .protection import main
import logging
import sys
import os
logging.getLogger('tensorflow').disabled = True


__all__ = (
    '__version__',
    'FawkesMaskGeneration', 'load_extractor',
    'init_gpu',
    'select_target_label', 'dump_image', 'reverse_process_cloaked', 'Faces', 'main'
)