#!/usr/bin/env python3
"""
Set up configs based on config range file

author: jy
"""


class Maker(object):
    """
    docstring for Maker.
    """

    def __init__(self, range_config:dict = None):
        super(Maker, self).__init__()
        self.configs = dict()
    #
    # def make_encoders(self, config):
