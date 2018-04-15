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

import ppad


# 3. Learning.
ppad.agent02.estimator.predict()

# 4. Predict and visualization.
for _ in range(100):
    observation = env.reset()
    for _ in range(2):
        action = agent.action(observation)
        observation, _, _, _ = env.step(action=action)
    print(env.actions)



