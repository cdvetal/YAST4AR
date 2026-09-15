#!/usr/bin/env python3
"""
Compatibility shim: some configs call `attacks/AutoAttack/AutoAttack.py`.
This file simply delegates execution to `run_autoattack.py` in the same folder.
"""
import runpy
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(this_dir, 'run_autoattack.py')

if __name__ == '__main__':
    runpy.run_path(script, run_name='__main__')
