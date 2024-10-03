from pathlib import Path
import time
from tmrl import get_environment
from time import sleep
import numpy as np
from model import Agent, Trainer
import torch
import nn_utilities as nn 

"""
This will be our policy given some LIDAR observations. 
    Input observations are of shape: ((1,), (4, 19), (3,), (3,))
        - Respresents: (speed, 4 last LIDARs, 2 previous actions)
    Returned action is: [gas, break, steer] analog between -1.0 and +1.0
"""

# ==========================================================================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================================================================================================

# initialize model1, model2, optimizer, and scaler
model1 = torch.jit.script(
    Agent(
        float_inputs_dim=nn.float_input_dim,
        float_hidden_dim=nn.float_hidden_dim,
        dense_hidden_dimension=nn.dense_hidden_dimension,
        iqn_embedding_dimension=nn.iqn_embedding_dimension,
        n_actions=len(nn.inputs)
    )
).to("cuda", memory_format=torch.channels_last)

model2 = torch.jit.script(
    Agent(
        float_inputs_dim=nn.float_input_dim,
        float_hidden_dim=nn.float_hidden_dim,
        dense_hidden_dimension=nn.dense_hidden_dimension,
        iqn_embedding_dimension=nn.iqn_embedding_dimension,
        n_actions=len(nn.inputs)
    )
).to("cuda", memory_format=torch.channels_last)

optimizer1 = torch.optim.RAdam(model1.parameters(), lr=nn.learning_rate, eps=1e-4)
scaler = torch.cuda.amp.GradScaler()

# setup directory where model weights are saved
base_dir = Path(__file__).resolve().parents[1]
run_name = "current_run_weights"
save_dir = base_dir / "save" / run_name
save_dir.mkdir(parents=True, exist_ok=True)

# load in current weights for model1 and model2
try:
    model1.load_state_dict(torch.load(save_dir / "weights1.torch"))
    model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
    optimizer1.load_state_dict(torch.load(save_dir / "optimizer1.torch"))
    print(" =========================     Weights loaded !     ================================")
    #nn.init_wandb(resume=True)
except:
    print(" Could not load weights")
    #nn.init_wandb(resume=False)

# start tracking loss and training durations
loss_history = []
train_duration_history = []
least_loss = 999999 # ATTENTION!!!!!! THIS MUST BE CHANGED EVERY TIME YOU RERUN!!!!!!!! otherwise files get overwritten
least_reward = 9999 # 1.31e+03

# initialize trainer
trainer = Trainer(
    model=model1,
    model2=model2,
    optimizer=optimizer1,
    scaler=scaler,
    batch_size=nn.batch_size,
    iqn_k=nn.iqn_k,
    iqn_n=nn.iqn_n,
    iqn_kappa=nn.iqn_kappa,
    epsilon=nn.epsilon,
    epsilon_boltzmann=nn.epsilon_boltzmann,
    gamma=nn.gamma,
    tau_epsilon_boltzmann=nn.tau_epsilon_boltzmann,
    tau_greedy_boltzmann=nn.tau_greedy_boltzmann,
)


model1.train()

# Let us retrieve the TMRL Gymnasium environment.
# The environment you get from get_environment() depends on the content of config.json
env = get_environment()

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

# ===============================================
#   TRAIN
# ===============================================

train_loop_start_time = time.time()

# -- training loop start --
i = 0

while True: 
    i += 1
    print(f"Start episode #{i}")

    # Only reset env if first episode
    if i == 1:
        obs, _ = env.reset()

        obs = np.concatenate((obs[0].flatten(), obs[1].flatten(), np.array([nn.continuous_to_discrete_action(obs[2].flatten())]),
                            np.array([nn.continuous_to_discrete_action(obs[3].flatten())])))
        trainer.cur_state = obs #torch.from_numpy(obs).cpu().float().to(device)

    # track time before training
    train_start_time = time.time()

    # train() gets action from model based on quantiles
    # converts discrete action_idx to tmrl continuous action and calls env.step(action)
    loss, reward, timesteps = trainer.train(env, do_learn=True)
    #loss, reward, timesteps, std_input, avg_input = trainer.train(env, do_learn=True)
    training_time = time.time() - train_start_time

    nn.record_wandb(loss, reward, training_time, timesteps)

    # save training time and loss
    # train_duration_history.append(time.time() - train_start_time)
    # loss_history.append(loss)
    print(f"B    {loss=:<8.2e},  {reward=:<8.2e}")

    nn.custom_weight_decay(model1, 1 - nn.weight_decay)

    # update target network if least loss
    if reward > least_reward:
        least_loss = loss
        least_reward = reward
        # ===============================================
        #   UPDATE TARGET NETWORK
        # ===============================================
        print("Updating target network")
        
        #nn.soft_copy_param(model2, model1, nn.soft_update_tau)
        # model2.load_state_dict(model.state_dict())

        # ===============================================
        #   SAVE WEIGHTS TO FILE
        # ===============================================
        #sub_folder_name = "least_loss_run"
        #(save_dir / "best_runs" / sub_folder_name).mkdir(parents=True, exist_ok=True)
        print("Saving weights")
        torch.save(
            model1.state_dict(),
            save_dir / "weights1.torch",
        )
        torch.save(
            model2.state_dict(),
            save_dir / "weights2.torch",
        )
        torch.save(
            optimizer1.state_dict(),
            save_dir / "optimizer1.torch", #"best_runs" / sub_folder_name / "optimizer1.torch",
        )

        # print(f"AVERAGE INPUT IS:")
        # print(avg_input)
        # print(f"STD INPUT IS:")
        # print(std_input)

    if (i % 50 == 0):
        train_loop_end_time = time.time()
        train_loop_minutes = (train_loop_end_time - train_loop_start_time) / 60
        print("Trained ", i, " episodes, which took ", train_loop_minutes, " minutes.\n")

    # reset loss and duration histories
    # loss_history = []
    # train_duration_history = []

    model1.train()


# -- training loop end --


# ==========================================================================================================================

# # Training loop
# num_episodes = 10  # Set the number of training episodes

# for episode in range(num_episodes):
#     print(f"\nEpisode: {episode + 1}\n")
    
#     # Reset the environment for each episode
#     obs, info = env.reset()

#     for step in range(200): 
#         # Get the action from the model
#         action_idx, _, _, _ = trainer.get_exploration_action(obs)

#         # Convert the action index to the corresponding action
#         action = inputs[action_idx]

#         # Take a step in the environment
#         obs, rew, terminated, truncated, info = env.step(action)

#         # Check if the episode is terminated
#         if terminated or truncated:
#             break

#     # Train the model after each episode
#     total_loss = trainer.train(env, do_learn=True)
#     print(f"Total Loss: {total_loss.item()}")

#     # Save the model if it has the least loss 
#     if episode == 0 or total_loss < best_loss:
#         best_loss = total_loss
#         torch.save(model1.state_dict(), "best_model.pth")





# ===============================================
#   SAVE STUFF IF THIS WAS A GOOD RACE
# ===============================================

# if end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"]:
#     # This is a new alltime_minimum

#     accumulated_stats["alltime_min_ms"] = end_race_stats["race_time"]

#     sub_folder_name = f"{end_race_stats['race_time']}"
#     (save_dir / "best_runs" / sub_folder_name).mkdir(parents=True, exist_ok=True)
#     joblib.dump(
#         rollout_results["actions"],
#         save_dir / "best_runs" / sub_folder_name / f"actions.joblib",
#     )
#     joblib.dump(
#         rollout_results["q_values"],
#         save_dir / "best_runs" / sub_folder_name / f"q_values.joblib",
#     )
#     torch.save(
#         model1.state_dict(),
#         save_dir / "best_runs" / sub_folder_name / "weights1.torch",
#     )
#     torch.save(
#         model2.state_dict(),
#         save_dir / "best_runs" / sub_folder_name / "weights2.torch",
#     )
#     torch.save(
#         optimizer1.state_dict(),
#         save_dir / "best_runs" / sub_folder_name / "optimizer1.torch",
#     )
# if end_race_stats["race_time"] < misc.good_time_save_all_ms:
#     sub_folder_name = f"{end_race_stats['race_time']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
#     (save_dir / "good_runs" / sub_folder_name).mkdir(parents=True, exist_ok=True)
#     joblib.dump(
#         rollout_results["actions"],
#         save_dir / "good_runs" / sub_folder_name / f"actions.joblib",
#     )
#     joblib.dump(
#         rollout_results["q_values"],
#         save_dir / "good_runs" / sub_folder_name / f"q_values.joblib",
#     )
#     torch.save(
#         model1.state_dict(),
#         save_dir / "good_runs" / sub_folder_name / "weights1.torch",
#     )
#     torch.save(
#         model2.state_dict(),
#         save_dir / "good_runs" / sub_folder_name / "weights2.torch",
#     )
#     torch.save(
#         optimizer1.state_dict(),
#         save_dir / "good_runs" / sub_folder_name / "optimizer1.torch",
#     )