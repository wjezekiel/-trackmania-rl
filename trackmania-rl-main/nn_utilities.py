import torch
import numpy as np

import logging
import os
import wandb
import random
import atexit

def init_kaiming(layer):
    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in")
    torch.nn.init.zeros_(layer.bias)

def custom_weight_decay(target_link, decay_factor):
    target_dict = target_link.state_dict()
    for k, target_value in target_dict.items():
        target_value.mul_(decay_factor)

def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of source link to target link."""
    target_dict = target_link.state_dict()
    source_dict = source_link.state_dict()
    for k, target_value in target_dict.items():
        source_value = source_dict[k]
        if source_value.dtype in [torch.float32, torch.float64, torch.float16]:
            assert target_value.shape == source_value.shape
            target_value.mul_(1 - tau)
            target_value.add_(tau * source_value)
        else:
            # Scalar type
            # Some modules such as BN has scalar value `num_batches_tracked`
            target_dict[k] = source_value
            assert False, "Soft scalar update should not happen"

def discrete_to_continuous_action(action_idx):
      if action_idx == 0: # Forward
          return np.array([1.0, 0.0, 0.0])
          
      elif action_idx == 1: # Forward left
          return np.array([1.0, 0.0, -1.0])
          
      elif action_idx == 2: # Forward right
          return np.array([1.0, 0.0, 1.0])

      elif action_idx == 3: # Nothing
          return np.array([0.0, 0.0, 0.0])
          
      elif action_idx == 4: # Nothing left
          return np.array([0.0, 0.0, -1.0])
          
      elif action_idx == 5: # Nothing right
          return np.array([0.0, 0.0, 1.0])
          
      elif action_idx == 6: # Brake
          return np.array([0.0, 1.0, 0.0])
          
      elif action_idx == 7: # Brake left
          return np.array([0.0, 1.0, -1.0])
          
      elif action_idx == 8: # Brake right
          return np.array([0.0, 1.0, 1.0])
          
      elif action_idx == 9: # Brake and accelerate
          return np.array([1.0, 1.0, 0.0])
          
      elif action_idx == 10: # Brake and accelerate left
          return np.array([1.0, 1.0, -1.0])
          
      else: #elif action_idx == 11: # Brake and accelerate right
          return np.array([1.0, 1.0, 1.0])
      
def continuous_to_discrete_action(action):
      if np.array_equal(action,np.array([1.0, 0.0, 0.0])): # Forward
          return 0
          
      elif np.array_equal(action,np.array([1.0, 0.0, -1.0])): # Forward left
          return 1
          
      elif np.array_equal(action,np.array([1.0, 0.0, 1.0])): # Forward right
          return 2

      elif np.array_equal(action,np.array([0.0, 0.0, 0.0])): # Nothing
          return 3
          
      elif np.array_equal(action,np.array([0.0, 0.0, -1.0])): # Nothing left
          return 4
          
      elif np.array_equal(action,np.array([0.0, 0.0, 1.0])): # Nothing right
          return 5
          
      elif np.array_equal(action,np.array([0.0, 1.0, 0.0])): # Brake
          return 6
          
      elif np.array_equal(action,np.array([0.0, 1.0, -1.0])): # Brake left
          return 7
          
      elif np.array_equal(action,np.array([0.0, 1.0, 1.0])): # Brake right
          return 8
          
      elif np.array_equal(action,np.array([1.0, 1.0, 0.0])): # Brake and accelerate
          return 9
          
      elif np.array_equal(action,np.array([1.0, 1.0, -1.0])): # Brake and accelerate left
          return 10
          
      else: #elif np.array_equal(action,np.array([1.0, 1.0, 1.0])): # Brake and accelerate right
          return 11

n_zone_centers_in_inputs = 40
float_input_dim = 1 + (4*19) + 1 + 1 # (the len of the flattened input array of size ((1,), (4, 19), (1,), (1,))) (79 vs 142)
float_hidden_dim = 256
dense_hidden_dimension = 64
iqn_embedding_dimension = 64
learning_rate = 1e-4

batch_size = 256
iqn_k = 32
iqn_n = 8
iqn_kappa = 1
epsilon = 0.01
epsilon_boltzmann = 0.1
gamma = 0.99
tau_epsilon_boltzmann = 0.1
tau_greedy_boltzmann = 0.01
weight_decay = 1e-5
soft_update_tau = 0.01
num_episodes = 200

inputs = [
    {  # 0 Forward
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 1 Forward left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 2 Forward right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": False,
    },
    {  # 3 Nothing
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 4 Nothing left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 5 Nothing right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": False,
    },
    {  # 6 Brake
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 7 Brake left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 8 Brake right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": True,
    },
    {  # 9 Brake and accelerate
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 10 Brake and accelerate left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 11 Brake and accelerate right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": True,
    },
]

# =========  WANDB CONFIG  ===========
wandb_run_id = "temp_for_recording_run"  # name for your run
#wandb_resume = True #"checkpoints"
wandb_project = "trackmania"  # name of the wandb project in which your run will appear
wandb_entity = "neeraja10"  # wandb account
wandb_key = "1fda8441e6e7ee859f59dc7743ce68725fc67161"  # wandb API key

# view graphs at https://wandb.ai/neeraja10/trackmania

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key

# =========  WANDB FUNCTIONS  ===========

def init_wandb(resume=False):
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        id=wandb_run_id,
        resume=resume
    )

def record_wandb(loss, reward, training_time, timesteps):
    # log metrics to wandb, creates 4 respective graphs, logs values each epoch
    wandb.log({"total_loss": loss, "total_reward": reward, "training_time": training_time, "num_timesteps": timesteps})

avg_input = [ 11.6230, 433.0969, 427.7646, 410.3868, 299.1592, 233.7096, 193.8670,
        169.4651, 153.8002, 143.5279, 137.4650, 133.9044, 135.0328, 139.3820,
        148.3500, 163.3015, 187.2362, 233.6344, 273.2726, 290.4419, 432.7328,
        426.5907, 409.2127, 298.3134, 233.0575, 193.3025, 168.9384, 153.3232,
        143.0426, 136.9790, 133.4057, 134.4762, 138.8125, 147.8166, 162.6745,
        186.4002, 232.4613, 272.1244, 289.7507, 432.1378, 425.4179, 408.0346,
        297.4661, 232.4022, 192.7413, 168.4103, 152.8456, 142.5597, 136.4913,
        132.9059, 133.9195, 138.2430, 147.2841, 162.0484, 185.5660, 231.2932,
        270.9783, 288.9074, 431.5700, 424.1431, 406.8087, 296.5890, 231.7222,
        192.1658, 167.8721, 152.3611, 142.0714, 135.9972, 132.3989, 133.3550,
        137.6573, 146.6422, 161.4111, 184.7121, 230.0825, 269.7409, 288.0845,
          3.2586,   3.2612]

std_input = [  8.9471,  91.7590, 116.4994, 127.2164,  92.8665,  77.1125,  64.1563,
         56.9309,  52.8144,  50.7812,  50.4397,  52.5145,  57.7461,  64.3247,
         75.4223,  91.6083, 113.1699, 155.7189, 177.2389, 177.3598,  92.3564,
        118.1270, 128.9600,  94.2602,  78.3108,  65.1685,  57.7374,  53.5921,
         51.4176,  50.9140,  52.9205,  58.0692,  64.6338,  75.9593,  92.0927,
        113.4604, 155.8130, 177.2617, 177.3340,  93.4109, 119.7203, 130.6710,
         95.6244,  79.4890,  66.1845,  58.5452,  54.3679,  52.0579,  51.3818,
         53.3188,  58.3853,  64.9320,  76.4859,  92.5675, 113.7398, 155.8941,
        177.2810, 177.3697,  94.3055, 121.5428, 132.4651,  97.0312,  80.6963,
         67.2139,  59.3622,  55.1351,  52.6880,  51.8442,  53.7113,  58.6953,
         65.2376,  76.7542,  93.0525, 114.0395, 156.0173, 177.3895, 177.3923,
          3.2810,   3.2882]