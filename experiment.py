import warnings
import torch
from omnisafe.common.experiment_grid import ExperimentGrid
import omnisafe
import os, sys


eg = ExperimentGrid(exp_name='TransferLearningExperiments')

# Set the algorithms.
policies = ['PPO', 'TRPO', 'PPOLag', 'TRPOLag']

# Set the environments.
leve1_safety_envs = [
    'SafetyPointGoal1-v0',
    'SafetyPointButton1-v0'
]
eg.add('env_id', leve1_safety_envs)
eg.add('algo', policies)
eg.add('logger_cfgs:use_wandb', [True])
eg.add('logger_cfgs:wandb_project', ['TransferLearningExperiments'])
eg.add('train_cfgs:vector_env_nums', [4])
eg.add('train_cfgs:torch_threads', [8])
eg.add('algo_cfgs:steps_per_epoch', [20000])
eg.add('train_cfgs:total_steps', [20000*30])
eg.add('seed', [42])

avaliable_gpus = list(range(torch.cuda.device_count()))
gpu_id = [0]

if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
    warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
    gpu_id = None

from omnisafe.typing import NamedTuple, Tuple
import omnisafe

def train(
    exp_id: str, algo: str, env_id: str, custom_cfgs: NamedTuple
) -> Tuple[float, float, float]:
    terminal_log_name = 'terminal.log'
    error_log_name = 'error.log'
    if 'seed' in custom_cfgs:
        terminal_log_name = f'seed{custom_cfgs["seed"]}_{terminal_log_name}'
        error_log_name = f'seed{custom_cfgs["seed"]}_{error_log_name}'
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
        os.makedirs(custom_cfgs['logger_cfgs']['log_dir'], exist_ok=True)
    # pylint: disable-next=consider-using-with
    sys.stdout = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', terminal_log_name),
        'w',
        encoding='utf-8',
    )
    # pylint: disable-next=consider-using-with
    sys.stderr = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', error_log_name),
        'w',
        encoding='utf-8',
    )
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len

def run_training(policies, envs, exp_name, wandb_project_name):
    eg = ExperimentGrid(exp_name=exp_name)
    eg.add('env_id', envs)
    eg.add('algo', policies)
    eg.add('logger_cfgs:use_wandb', [True])
    eg.add('logger_cfgs:wandb_project', [wandb_project_name])
    eg.add('train_cfgs:vector_env_nums', [4])
    eg.add('train_cfgs:torch_threads', [8])
    eg.add('algo_cfgs:steps_per_epoch', [20000])
    eg.add('train_cfgs:total_steps', [20000*30])
    eg.add('seed', [42])

    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None
    
    eg.run(train, gpu_id=gpu_id)

def evaluate_exp(log_dir):
    LOG_DIR = 'exp-x\TransferModels\SafetyPointGoal1-v0---f58ca1a463c0273cc1b6ca3902826cae538e180bd08f94f50e82a46844a78ecc\TRPOLag-{SafetyPointGoal1-v0}\seed-000-2024-05-19-17-50-17'
    evaluator = omnisafe.Evaluator()
    result = None
    for item in os.scandir(os.path.join(log_dir, 'torch_save')):
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved(
                save_dir=LOG_DIR, model_name=item.name, camera_name='track', width=256, height=256
            )
            # evaluator.render(num_episodes=5)
            result = evaluator.evaluate(num_episodes=50)