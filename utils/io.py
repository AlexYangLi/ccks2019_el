# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: io.py

@time: 2019/4/29 16:00

@desc:

"""

import os
import json
import pickle
from config import SUBMIT_DIR


def pickle_load(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        print('Logging Info - Loaded:', filename)
    except EOFError:
        print('Logging Error - Cannot load:', filename)
        obj = None

    return obj


def pickle_dump(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

    print('Logging Info - Saved:', filename)


def write_log(filename, log, mode='w'):
    with open(filename, mode) as writer:
        writer.write('\n')
        json.dump(log, writer, indent=4, ensure_ascii=False)


def format_filename(_dir, filename_template, **kwargs):
    """Obtain the filename of data base on the provided template and parameters"""
    filename = os.path.join(_dir, filename_template.format(**kwargs))
    return filename


def submit_result(submit_file, test_data, pred_result):
    try:
        assert len(test_data) == len(pred_result)
        with open(os.path.join(SUBMIT_DIR, submit_file), 'w') as writer:
            for i in range(len(test_data)):
                text_id = test_data[i]['text_id']
                text = test_data[i]['raw_text']
                mention_data = [{'mention': text[x[1]: x[1]+len(x[0])], 'offset': str(x[1]), 'kb_id': x[2]}
                                for x in pred_result[i]]    # 评测结果文件中mention名必须和输入文本保持一致
                submit = {'text_id': text_id, 'text': text, 'mention_data': mention_data}
                # http://litaotju.github.io/python/2016/06/28/python-json-dump-utf/
                json.dump(submit, writer, ensure_ascii=False)
                writer.write('\n')
        print('Logging Info - Generate test result in {}'.format(submit_file))
    except OSError:
        print('file name too long, will use `results.json` instead')
        submit_result('results.json', test_data, pred_result)
