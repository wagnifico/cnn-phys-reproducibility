#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities functions
"""

# native modules
import functools

# third-party modules

# local modules

def calltracker(func):
    """
        A decorator to check whether a class method has been called.
    """
    @functools.wraps(func)
    def wrapper(*args):
        wrapper._called = True
        return func(*args)
    wrapper._called = False
    return wrapper


