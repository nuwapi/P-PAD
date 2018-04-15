"""
P-PAD
Copyright (C) 2018 NWP, CP

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

import logging
import os
import sys

# Set up logger.
logging_root = logging.getLogger()
# Set up logger format.
logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s')

# Set up system environment variables.
os.environ['PPADPATH'] = os.path.dirname(os.path.abspath(__file__)) + '/..'
logging.info('Configured PPADPATH to {0}'.format(os.environ['PPADPATH']))
if 'PYTHONPATH' not in os.environ:
    os.environ['PYTHONPATH'] = os.environ['PPADPATH']
    sys.path.append(os.environ['PYTHONPATH'])
elif os.environ['PYTHONPATH'].find('P-PAD') == -1:
    os.environ['PYTHONPATH'] = os.path.join(os.environ['PYTHONPATH'], os.environ['PPADPATH'])
    sys.path.append(os.environ['PYTHONPATH'])
logging.info('Added PPADPATH to PYTHONPATH. Your current PYTHONPATH is {0}'.format(os.environ['PYTHONPATH']))

# NOTE: ppad.agent and ppad.pad are the central modules of P-PAD.
# Within P-PAD, ppad.* modules can only import ppad.agent and ppad.pad but not any other ppad.*,
# while ppad.agent and ppad.pad are each completely independent and self-sufficient modules.
from ppad.agent.agent01 import agent01
import ppad.agent.agent02 as agent02
from ppad.agent.utils import discount
from ppad.pad.game import PAD
from ppad.data.dataset01.smart_data import smart_data
