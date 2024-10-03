import math
import random
from typing import Optional, Tuple

import numpy as np
import torch

import nn_utilities as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim,       
        float_hidden_dim,        
        dense_hidden_dimension,  
        iqn_embedding_dimension, 
        n_actions,               
    ):
        super().__init__()
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.A_head = torch.nn.Sequential(
            torch.nn.Linear(float_hidden_dim, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, n_actions),
        )
        self.V_head = torch.nn.Sequential(
            torch.nn.Linear(float_hidden_dim, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, 1),
        )

        self.iqn_fc = torch.nn.Linear(
            iqn_embedding_dimension, float_hidden_dim
        )  # There is no word in the paper on how to init this layer?
        self.lrelu = torch.nn.LeakyReLU()
        self.initialize_weights()

        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.n_actions = n_actions

        self.avg_input = torch.from_numpy(np.array(nn.avg_input)).cpu().float().to(device)
        self.std_input = torch.from_numpy(np.array(nn.std_input)).cpu().float().to(device)

    def initialize_weights(self):
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                nn.init_kaiming(m)
        # This was uninitialized in Agade's code
        nn.init_kaiming(self.iqn_fc)
        # A_head and V_head are NoisyLinear, already initialized

    def forward(self, float_inputs, num_quantiles: int = 8, tau: Optional[torch.Tensor] = None,
                 use_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            float_inputs: input of shape (batch_size, (1,), (4, 19), (3,), (3,))
                representing speed, past 4 LIDARS, past 2 actions
            num_quantiles: int, typically 8, which has a dramatic early performance boost
                but not worth increasing past 8 (has minimal performance improvements)
            tau: not really, just ignore I guess
        Returns:
            (Q, tau)
        """
        batch_size = float_inputs.shape[0]
        float_outputs = self.float_feature_extractor((float_inputs - self.avg_input) / self.std_input)
        # (batch_size, float_hidden_dim) OK
        if tau is None:
            tau = torch.rand(
                size=(batch_size * num_quantiles, 1), device="cuda", dtype=torch.float32
            )  # (batch_size * num_quantiles, 1) (random numbers)
        quantile_net = tau.expand(
            [-1, self.iqn_embedding_dimension]
        )  # (batch_size*num_quantiles, iqn_embedding_dimension) (still random numbers)
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device="cuda") * math.pi * quantile_net
        )  # (batch_size*num_quantiles, iqn_embedding_dimension)
        # (8 or 32 initial random numbers, expanded with cos to iqn_embedding_dimension)
        # (batch_size*num_quantiles, float_hidden_dim)
        quantile_net = self.iqn_fc(quantile_net)
        # (batch_size*num_quantiles, float_hidden_dim)
        quantile_net = self.lrelu(quantile_net)
        # (batch_size*num_quantiles, float_hidden_dim)
        float_outputs = float_outputs.repeat(num_quantiles, 1)
        # (batch_size*num_quantiles, float_hidden_dim)
        float_outputs = float_outputs * quantile_net

        A = self.A_head(float_outputs)  # (batch_size*num_quantiles, n_actions)
        V = self.V_head(float_outputs)  # (batch_size*num_quantiles, 1) #need to check this

        Q = V + A - A.mean(dim=-1).unsqueeze(-1)

        return Q, tau

class Trainer:
    def __init__(
        self,
        model: Agent,
        model2: Agent,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
        batch_size: int,
        iqn_k: int,
        iqn_n: int,
        iqn_kappa: float,
        epsilon: float,
        epsilon_boltzmann: float,
        gamma: float,
        tau_epsilon_boltzmann: float,
        tau_greedy_boltzmann: float,
    ):
        self.model = model
        self.model2 = model2
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.iqn_k = iqn_k
        self.iqn_n = iqn_n
        self.iqn_kappa = iqn_kappa
        self.epsilon = epsilon
        self.epsilon_boltzmann = epsilon_boltzmann
        self.gamma = gamma
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.tau_greedy_boltzmann = tau_greedy_boltzmann
        self.cur_state = None
        #self.input_tensor_total = []

    def train(self, env, do_learn: bool):
        self.optimizer.zero_grad(set_to_none=True)
        terminate = False

        total_loss = 0
        total_reward = 0
        iters = 0

        while (not terminate):
            iters += 1
            state_float_tensor = [] # Current states (S)
            new_actions = []
            new_n_steps = 4
            rewards = []
            next_state_float_tensor = [] # Next states (S')
            gammas_per_n_steps = []
            new_done = []

            for i in range(new_n_steps):
                #self.input_tensor_total.append(self.cur_state)
                state_float_tensor.append(self.cur_state)
                float_input = torch.from_numpy(np.expand_dims(self.cur_state, axis=0)).cpu().float().to(device)
                action_idx, _, _, _ = self.get_exploration_action(float_input)
            
                new_actions.append(action_idx)

                act = nn.discrete_to_continuous_action(action_idx)

                obs, rew, done, truncated, info = env.step(act)  # step (rtgym ensures healthy time-steps)

                obs = np.concatenate((obs[0].flatten(), obs[1].flatten(), np.array([nn.continuous_to_discrete_action(obs[2].flatten())]),
                            np.array([nn.continuous_to_discrete_action(obs[3].flatten())])))
                self.cur_state = obs #torch.from_numpy(obs).cpu().float().to(device)

                next_state_float_tensor.append(self.cur_state)
                #if iters == 1 and done:
                #    rew -= new_n_steps/(i+1)
                if done and iters < 50:
                    rew -= 1/(iters)
                rew -= 0.01 # punishment for taking too long
                if iters < 12 and action_idx in (6,7,8,9,10,11):
                    rew -= 0.1
                rewards.append(rew)
                new_done.append(done)

                gammas_per_n_steps.append(self.gamma)
                self.gamma *= self.gamma

                if done:
                    terminate = True

                    obs, _ = env.reset()
                    obs = np.concatenate((obs[0].flatten(), obs[1].flatten(), np.array([nn.continuous_to_discrete_action(obs[2].flatten())]),
                                        np.array([nn.continuous_to_discrete_action(obs[3].flatten())])))
                    self.cur_state = obs
                    
                    break

            self.batch_size = i+1
                
            state_float_tensor = torch.from_numpy(np.array(state_float_tensor)).cpu().float().to(device)
            new_actions = torch.from_numpy(np.array(new_actions)).cpu().to(device)
            new_n_steps = torch.tensor(new_n_steps, dtype=torch.int64)
            rewards = torch.from_numpy(np.array(rewards)).cpu().float().to(device)
            next_state_float_tensor = torch.from_numpy(np.array(next_state_float_tensor)).cpu().float().to(device)
            gammas_per_n_steps = torch.from_numpy(np.array(gammas_per_n_steps)).cpu().float().to(device)
            new_done = torch.from_numpy(np.array(new_done)).cpu().to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                with torch.no_grad():
                    new_actions = new_actions.to(dtype=torch.int64)
                    new_n_steps = new_n_steps.to(dtype=torch.int64)

                    rewards = rewards.reshape(-1,1).repeat([self.iqn_n, 1]) # (batch_size*iqn_n, 1)
                    gammas_pow_nsteps = gammas_per_n_steps.reshape(-1, 1).repeat([self.iqn_n, 1]) # (batch_size*iqn_n, 1)
                    done = new_done.reshape(-1, 1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                    actions = new_actions[:, None]  # (batch_size, 1)
                    actions_n = actions.repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                    #
                    #   Use model to choose an action for next state.
                    #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
                    #
                    a__tpo__model__reduced_repeated = (
                        self.model(
                            next_state_float_tensor,
                            self.iqn_n,
                            tau=None,
                        )[0]
                        .reshape([self.iqn_n, self.batch_size, self.model.n_actions])
                        .mean(dim=0)
                        .argmax(dim=1, keepdim=True)
                        .repeat([self.iqn_n, 1])
                    )  # (iqn_n * batch_size, 1)

                    #
                    #   Use model2 to evaluate the action chosen, per quantile.
                    #
                    q__stpo__model2__quantiles_tau2, tau2 = self.model2(
                        next_state_float_tensor, self.iqn_n, tau=None
                    )  # (batch_size*iqn_n,n_actions)

                    # print(f"a__tpo: {a__tpo__model__reduced_repeated.shape}")
                    # print(f"q__stpo: {q__stpo__model2__quantiles_tau2.shape}")
                    # print(f"gammas: {gammas_pow_nsteps.shape}")
                    # print(f"rewards: {rewards.shape}")
                    # print(f"done: {done.shape}")

                    #
                    #   Build IQN target on tau2 quantiles
                    #
                    outputs_target_tau2 = torch.where(
                        done,
                        rewards, 
                        rewards + gammas_pow_nsteps * q__stpo__model2__quantiles_tau2.gather(1, a__tpo__model__reduced_repeated),
                        # rewards + gammas_pow_nsteps * q__stpo__model2__quantiles_tau2.gather(1, a__tpo__model__reduced_repeated),
                    )  # (batch_size*iqn_n, 1)

                    #
                    #   This is our target
                    #
                    outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                        0, 1
                    )  # (batch_size, iqn_n, 1)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                q__st__model__quantiles_tau3, tau3 = self.model(
                    state_float_tensor, self.iqn_n, tau=None
                )  # (batch_size*iqn_n,n_actions)

                outputs_tau3 = (
                    q__st__model__quantiles_tau3.gather(1, actions_n).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
                )  # (batch_size, iqn_n, 1)

                TD_Error = outputs_tau3[:, None, :, :] - outputs_target_tau2[:, :, None, :]
                # (batch_size, iqn_n, iqn_n, 1) 
                # Huber loss, my alternative
                loss = torch.where(
                    torch.abs(TD_Error) <= self.iqn_kappa,
                    0.5 * TD_Error**2,
                    self.iqn_kappa * (torch.abs(TD_Error) - 0.5 * self.iqn_kappa),
                )
                tau3 = tau3.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
                tau3 = tau3[:, None, :, :].expand([-1, self.iqn_n, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
                loss = (
                    (torch.where(TD_Error < 0, 1 - tau3, tau3) * loss / self.iqn_kappa).sum(dim=2).mean(dim=1)[:, 0]
                )  # pinball loss # (batch_size, )

                loss = torch.sum(loss)  # total_loss.shape=torch.Size([])
                total_loss += loss
                #total_loss = loss

            if do_learn:
                #print("Updating model scaler")
                self.scaler.scale(loss).backward()

                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_reward += torch.sum(rewards)

        #total_loss /= iters
        #total_reward /= iters

        self.gamma = nn.gamma 

        total_loss = total_loss.detach().cpu().item()
        total_reward = total_reward.detach().cpu().item()
        timesteps = ((iters-1)*new_n_steps) + i

        #input_tensor_total = torch.from_numpy(np.array(self.input_tensor_total)).cpu().float().to(device)
        #std_input, avg_input = torch.std_mean(input_tensor_total, dim=0)

        return total_loss, total_reward, timesteps #, std_input, avg_input

    def get_exploration_action(self,float_inputs):
        with torch.no_grad():
            state_float_tensor = float_inputs #torch.as_tensor(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            q_values = (
                self.model(state_float_tensor, self.iqn_k, tau=None, use_fp32=True)[0]
                .cpu()
                .numpy()
                .astype(np.float32)
                .mean(axis=0)
            )
        r = random.random()

        if r < self.epsilon:
            # Choose a random action
            get_argmax_on = np.random.randn(*q_values.shape)
        elif r < self.epsilon + self.epsilon_boltzmann:
            get_argmax_on = q_values + self.tau_epsilon_boltzmann * np.random.randn(*q_values.shape)
        else:
            get_argmax_on = q_values + ((self.epsilon + self.epsilon_boltzmann) > 0) * self.tau_greedy_boltzmann * np.random.randn(
                *q_values.shape
            )

        action_chosen_idx = np.argmax(get_argmax_on)
        greedy_action_idx = np.argmax(q_values)

        return (
            action_chosen_idx,
            action_chosen_idx == greedy_action_idx,
            np.max(q_values),
            q_values,
        )
