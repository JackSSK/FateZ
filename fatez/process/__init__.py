#!/usr/bin/env python3
"""
This folder contains basic processes for FateZ.

author: jy
"""
import io
import sys

class Error(Exception):
    pass



class Quiet_Mode(object):
    """
    Capable to suppress sys std output (print function)
    """
    trap = io.StringIO()
    stdout = sys.stdout

    def on(self):
        sys.stdout = self.trap

    def off(self):
        sys.stdout = self.stdout

    def get_captured(self):
        return self.trap.getvalue()
