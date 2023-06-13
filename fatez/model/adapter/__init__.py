#!/usr/bin/env python3
"""
Low rank adaption related objects
Very experimental!


author: jy
"""
from fatez.model.adapter.lora import Model as LoRA

__all__ = [
    'LoRA',
]

class Error(Exception):
    pass
