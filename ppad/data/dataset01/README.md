# Dataset 01
Smart data version 1.

Summary
* 16 human-solved boards as the starting point.
  Each board has [a1, a2, a3, a4, a5, a6] number of orbs of each type and sum(ai) = 30.
  The distribution of the values of a1 to a6 roughly follow the skyfall rate where each orb type is equally probable.
* Trajectories are generated starting from the solved boards, where each step in the trajectories is randomly chosen.
* The trajectories are played backwards pretending that someone started from the end state and eventually solved the board using the reverse path.

To-do
* Decrease entropy for every step in the trajectory.