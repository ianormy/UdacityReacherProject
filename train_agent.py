"""Train a network

This uses an experiment framework that enables trying different
hyperparameters and storing the results.

It uses stable-baselines3 for the implementation of the
reinforcement learning algorithms.

At the moment only the TD3 algorithm is supported, which was
sufficient to solve the project specification of achieving
a mean reward above 30.0 for the last 100 episodes.
"""
import argparse
import sys
import random
import os

from unityml.reward_callback import RewardCallback
from unityml.unity_ml_facade import UnityMlFacade
from stable_baselines3.common.monitor import Monitor
import stable_baselines3.td3 as td3
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an agent using stable-baselines3 and Unity ML agents",
        epilog="This uses an old version of Unity ML agents ")
    parser.add_argument(
        '--environment-port',
        type=int,
        default=5005,
        required=False,
        help='Unity environment port to use for communicating with Unity ML agent')
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=1e6,
        required=False,
        help='Limit the total number of timesteps for training')
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='td3_lr_0_00003_128_128_128',
        required=False,
        help='name to use for the experiment')
    parser.add_argument(
        '--experiments-root',
        type=str,
        default='.',
        required=False,
        help='path to the experiments root folder')
    parser.add_argument(
        '--algorithm',
        type=str,
        default='td3',
        required=False,
        help='algorithm used for reinforcement learning')
    parser.add_argument(
        '--executable-path',
        type=str,
        default='D:/Source/Udacity/UdacityReacherProject/Reacher_Windows_x86_64/Reacher.exe',
        required=False,
        help='path to the Reacher executable')
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        required=False,
        help='seed to use for training. Default of -1 means use a random seed')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        required=False,
        help='Size of batches to use for training')
    parser.add_argument(
        '--policy-layers',
        type=str,
        default='400,300',
        required=False,
        help='policy hidden layers')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0003,
        required=False,
        help='learning rate')
    parser.add_argument(
        '--reward-threshold',
        type=float,
        default=30.0,
        required=False,
        help='Reward threshold for the mean of the last 100 episodes')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        required=False,
        help='Greedy strategy gamma value')
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1000000,
        required=False,
        help='Size of the buffer for replay memory')
    return parser, parser.parse_args()


def main():
    # parse arguments
    parser, args = parse_args()

    # exit and show help if no arguments provided at all
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # create all the folders needed for the experiment
    experiment_folder = os.path.join(args.experiments_root, args.experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    model_folder = os.path.join(experiment_folder, 'model')
    os.makedirs(model_folder, exist_ok=True)
    results_folder = os.path.join(experiment_folder, 'results')
    os.makedirs(results_folder, exist_ok=True)
    # create the environment
    env = UnityMlFacade(executable_path=args.executable_path, seed=args.seed,
                        environment_port=args.environment_port)
    # monitor the episode rewards and output them to a CSV file
    monitor_file = os.path.join(results_folder, "episode_results.monitor.csv")
    env = Monitor(env, monitor_file)
    # create the model agent using the parameters provided
    policy_layers = [int(layer_width) for layer_width in args.policy_layers.split(',')]
    seed = random.randint(0, int(1e6)) if args.seed == -1 else args.seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.algorithm == 'td3':
        model_name = 'TD3'
        policy_kwargs = dict(net_arch=policy_layers)
        model = td3.TD3(
            td3.MlpPolicy,
            env,
            verbose=1,
            device=device,
            gamma=args.gamma,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size)
    else:
        print("Sorry, only the TD3 algorithm is currently supported")
        sys.exit(-1)
    # setup the callback
    reward_callback = RewardCallback(
        eval_env=env,
        check_freq=500,
        reward_threshold=args.reward_threshold
    )
    # save all the hyperparameters for this experiment
    with open(os.path.join(experiment_folder, 'hyperparameters.txt'), 'w') as f:
        f.write('learning rate: {}\n'.format(args.learning_rate))
        f.write('policy layers: {}\n'.format(args.policy_layers))
        f.write('seed: {}\n'.format(seed))
        f.write('algorithm: {}\n'.format(args.algorithm))
        f.write('batch size: {}\n'.format(args.batch_size))
        f.write('buffer size: {}\n'.format(args.buffer_size))
        f.write('gamma: {}\n'.format(args.gamma))
        f.write('total timesteps: {}\n'.format(args.total_timesteps))
    # run the experiment
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[reward_callback]
    )
    # save the final model
    model.save(os.path.join(model_folder, model_name))


if __name__ == '__main__':
    main()
