This is code for reproducing the simulated experiments in the paper, "Learning Manipulation Skills Via Hierarchical Spatial Attention", by Marcus Gualtieri and Robert Platt. See my website, http://www.ccs.neu.edu/home/mgualti for details. The code is similar to that in the earlier 6-DoF version, https://github.com/mgualti/DeepRLManip, but this version is much faster and should be easier to get started with.

Getting started:
1) Pick a domain. From simplest to most complex: "Tabular Pegs on Disks", "Upright Pegs on Disks", "Pegs on Disks", and "Bottles on Coasters".
2) Non-tabular domains require Keras with Tensorflow (https://www.tensorflow.org/install/source), PointCloudsPython (https://github.com/mgualti/PointCloudsPython), and OpenRAVE (https://github.com/rdiankov/openrave). These instructions for installing OpenRAVE were helpful: https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html. Last time I checked, the main branch of OpenRAVE was sufficient. I plot the results using Matlab, but this is not a strict requirement.
3) For "Pegs on Disks" and "Bottles on Coasters", generate object meshes using the scripts named python/generate_*.py. Make sure to adjust the paths in the scripts. "Bottles on Coasters" requires 3DNet Cat-10, which can be downloaded from https://strands.readthedocs.io/en/latest/datasets/three_d_net.html.
4) From the domain's main directory, run "python/train.py train_params". This will run the experiment with the parameter file train_params.py. Some paths may need adjusted in the parameter file.
5) In the parameter file, adjust the visualization/plotting parameters to see if the simulation is working.

Highlights:
- Grasp and place checks, including the antipodal conditition are in the rl_environment_[domain].py class.
- HSA is implemented in the files starting with rl_agent.

Data:
Trained model files and result files are in a separate repository. Let me know if there is an interest.

Robot experiments:
The ROS nodes and motion planning code is all in a separte repository. Let me know if there is an interest in using this code as well.
