#!/usr/bin/env python3
"""
JSON file related tools

author: jy, nkmtmsys
"""
import re
import gzip
import json


def encode(data = None, save_path:str = 'out.js', indent:int = 4):
    """
    Encode data in json format file.

    :param data:
        Either dict() or list() data to be encoded.

    :param save_path: <str Default = 'out.js'>
        Path to save the output file.

    :param indent: <int Default = 4>
        Length of each indent.
    """
    if re.search(r'\.gz$', save_path):
        with gzip.open(save_path, 'w+') as output:
            output.write(json.dumps(data).encode('utf-8'))
    else:
        with open(save_path, 'w+') as output:
            json.dump(data, output, indent = indent)


def decode(path:str = None):
    """
    Decode data from JSON format file.

    :param path: <str Default = None>
        Path to the input file.
    """
    if re.search(r'\.gz$', path):
        with gzip.open(path, 'r') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        return data
