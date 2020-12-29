# Setup project
Here are notes about how to setup the project. 

Unfortunately, the Unity ML environment used by Udacity for this project is a very early version that is a few years old - [v0.4](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0). This makes it extremely difficult to set things up, particularly in a Windows environment. 


## Getting Started
To setup the project follow those steps:

1. Provide an environment with `python 3.6.x` installed - ideally by creating a new virtual environment. An example would be:

```
virtualenv drlnd_reacher -py 3.6
```

2. Clone and install the requirements of the project: 
```
git clone git@github.com:ianormy/UnityReacherProject.git
cd reacher-reinforcement-learning
pip install -r requirements.txt
```

3. Install a version of pytorch compatible with your architecture. The version used by this project is 1.5.0. This is an old version, but it's the newest that works with this old project :-) To use the correct version that is compatible with your CUDA (if you want to use GPU) then you will need the correct install string. Please see [this document](https://pytorch.org/get-started/previous-versions/) for help with this. My version of CUDA is 10.1 on Windows so I used this command:

```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install the unity environment provided by Udacity - **unityagents**. To help with that I have created a wheel that has everything in it. It's in the wheels folder of this repository. To install it simply do this:

```
pip install wheels\unityagents-0.4.0-py3-none-any.whl
```

5. Download and extract in the root of the project the environment compatible with your architecture:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
