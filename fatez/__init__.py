#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Copyright (C) 2022 Jack Yu <gyu17@jhu.edu>
# Author: Jack Yu <gyu17@jhu.edu> <Shenzhen Mozhou Technology Co., Ltd.>

# This program is free software
# You can redistribute it and/or modify it
# under the terms of the GNU General Public License
# as published by the Free Software Foundation;
# either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Fate_Zero(Temp_name)
========
Summary
-------
Some discrption here

For more information, please visit our GitHub repo:
URL here
--------

"""

__version__ = "0.0.1a5"
__author__ = "JackSSK"
__email__ = "gyu17@jhu.edu"

from ._main import (
    Launch
)

from .test import (
    Test
)

__all__ = [
    'Test',
    'Launch',
]
