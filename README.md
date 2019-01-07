# Twin_Delayed_DDPG
Pytorch Implementation of Twin Delayed DDPG

# Project Details

<ul>
  <li> The environment consists of agent where the task of the agent is an double-jointed arm and the goal of the agent is to maintain the arm at the target for as long as possible</li>
  <li> The current state of the environment is represented by 33 dimensional feature vector and corresponding to position, rotation, velocity, and angular velocities of the arm</li
  <li> Action space is continous and thus it represent by a vector with four numbers, corresponding to torque applicable to two joints ranging between -1 and 1 in each dimension.</li>
  <li> A reward of +0.1 is provided for each step the agent's hand is at target location.</li>
  <li> The task is episoidic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes</li>
</ul>

# Technical Dependencies

<ol>
  <li> Python 3.6 :
  <li> PyTorch (0.4,CUDA 9.0) : pip3 install torch torchvision</li>
  <li> ML-agents (0.4) : Refer to <a href = "https://github.com/Unity-Technologies/ml-agents/">ml-agents</a> for installation</li>
  <li> Numpy (1.14.5) : pip3 install numpy</li>
  <li> Matplotlib (3.0.2) : pip3 install matplotlib</li>
  <li> Jupyter notebook : pip3 install jupyter </li>
  <li> Download the environment from <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip">here</a> and place it in the same folder as that of CC.ipynb file  </li>
</ol>

# Network details

- [x] OUNoise
- [x] Delayed Policy Updates
- [x] Two Critic Networks

# Installation Instructions :
`
step 1 : Install all the dependencies
`
<br>
`
step 2 : git clone https://github.com/adithya-subramanian/Twin_Delayed_DDPG.git
`
<br>
`
step 3 : jupyter notebook
`
<br>
`
step 4 : Run all cells in the CC.ipynb file
`
# Acknowledgment

Certain parts of TD_3_agent.py,model.py and CC.ipynb has been partially taken from the Udacity's deep reinforcement learning Nanodegree.
