# Humanoid-RL
training our custom humanoid to stand and walk with various algorithms. PPO first

|   PPO  |   JAX  |
| :---: | :---: |
| [6m_steps.webm](https://gist.github.com/user-attachments/assets/b7c1c253-cb0a-4739-9ebd-7f2505230ba6) |[jax_colab](https://gist.github.com/user-attachments/assets/bfb1915e-ff0a-44fd-92b1-bcfe281622d1) |

## Lessons

### conversion issues

- everything was fixed in urdf and connected relative to general origin, apparently mjcf needs parent-child origin details
- was using position actuators not motor actuators(so they just hels one joint angle stiffly), base was fixed to the world, and gravity was off, so simulation wasn't working unless paused
- also had to align model's direction, ctrlrange, to match the stock humanoid cause I'm working off of a script that uses that

### Training PPO lessons

- if you don't specify your own xml file then env_id = humanoid-v5 deafults to mujoco's custom humanoid
- Robot had settled into a sort of backward half crouch at 5M steps, while moving forward and exploiting rewards, even though the tensor graphs looked fine. Had to stop the training at 25%(1 hour!!) andthe robot seems to be 
- added a penalty for if the robot height goes below 85% 
- then robot started using its toes at some point for the left leg, and turning towards the left
- added a penalty for not staying in the x direction to fix that, the toe problem didn't occur again in the next run
  
*- seems the bending stuff came back in the mjx training*

## General Training notes

PPO - about 3 hrs (for 6m steps, didn't complete 15) <br>
MJX - 1hr, 27min
