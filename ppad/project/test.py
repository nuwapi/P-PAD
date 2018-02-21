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

import numpy as np
import os

import ppad.pad.game as game

episode = []
actions = []
env = game.make()
board, finger = env.state()
episode.append(np.copy(board))

for i in range(100):
    action = env.action_space.sample()
    env.step(action)
    actions.append(action)
    episode.append(np.copy(env.state()[0]))

env.episode2gif(episode, actions, finger, os.environ['PYTHONPATH']+'/asset/sample.gif')

# env = game.make()
# env.render()
# env.step('left')
# env.render()
# env.step('up')
# env.render()

# env = game.make()
# env.render()
# observation, reward, done, info = env.step('pass', verbose=True)
# env.render()

# import os
# import numpy as np
# from PIL import Image
#
# asset_path = os.environ['PYTHONPATH'] + '/asset/'
# orbs = [asset_path + 'red.png',
#         asset_path + 'blue.png',
#         asset_path + 'green.png',
#         asset_path + 'light.png',
#         asset_path + 'dark.png',
#         asset_path + 'heal.png',
#         asset_path + 'jammer.png',
#         asset_path + 'poison.png',
#         asset_path + 'mortal_poison.png',
#         asset_path + 'bomb.png']
# bg = [asset_path + 'bg1.png', asset_path + 'bg2.png']
#
#
# print(os.environ['PYTHONPATH'])
#
#
#
#
# background = Image.open(bg[0])
# foreground = Image.open(orbs[0])
#
# background.paste(foreground, (0, 0), foreground)
#
# background2 = Image.open(bg[1])
# foreground2 = Image.open(orbs[1])
#
# background2.paste(foreground2, (0, 0), foreground2)
#
# result = Image.new("RGB", (800, 800))
#
# result.paste(background, (0, 0, 100, 100))
# result.paste(background2, (100, 0, 200, 100))
# result.paste(foreground, (150, 0, 250, 100), foreground)
#
# #result.show()
#
# im = Image.new('RGB', (400, 400))
# im.save(asset_path+'sample.gif', save_all=True, append_images=[background, background2], duration=2)
#
# print(0 % 2)
# print(1 % 2)
# print(2 % 2)