#!/usr/bin/env python
""" This is the python EML Toolbox.

EML Toolbox is a collection of routines and tools which I developed during my
work with the electromagnetic levitation setup. It contains mainly functions 
and routines that are used for edge-detection processing of images acquired 
during levitation experiments but also routines to process pyrometer log-files 
originating from the software used with LumaSense pyrometers. Moreover, one or
another useful python script is included that may be useful for other 
researchers in the field. 

BSD 3-Clause License

Copyright (c) 2020, Thomas Leitner
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Thomas Leitner"
__contact__ = "leitner.aut@gmail.com"
__copyright__ = "Copyright $YEAR"
__date__ = "2020/10/15"
__deprecated__ = False
__email__ =  "leitner.aut@gmail.com"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Thomas Leitner"
__status__ = "development"
__version__ = "0.1"

# docstring template according to: 
# https://gist.github.com/NicolasBizzozzero/6d4ca63f8482a1af99b0ed022c13b041

from . import _edge_detection as ed;
from . import _edge_fitting as ef;
from . import _various as various