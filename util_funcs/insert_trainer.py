import torch
import torch.nn.functional as F
from network_models.insert_model import QNetwork, hard_update, GaussianPolicy, soft_update
from torch.optim import Adam
import os


class SAC(object):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 gamma=0.9,
                 tau=0.005,
                 alpha=0.02,
                 optimizer_lr=1e-5,
                 target_update_interval=1,
                 q_net_hidden_size=128,
                 entropy_tuning=False):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = entropy_tuning
        self.q_net_hidden_size = q_net_hidden_size

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.critic = QNetwork(num_inputs,
                               num_actions,
                               self.q_net_hidden_size).to(device=self.device,
                                                          dtype=torch.float)
        self.critic_optim = Adam(self.critic.parameters(),
                                 lr=optimizer_lr)

        self.critic_target = QNetwork(num_inputs,
                                      num_actions,
                                      self.q_net_hidden_size).to(device=self.device,
                                                                 dtype=torch.float)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.tensor(num_actions).to(self.device)).item()
            # Check target_entropy
            self.log_alpha = torch.zeros(1,
                                         requires_grad=True,
                                         device=self.device)
            self.alpha_optim = Adam([self.log_alpha],
                                    lr=optimizer_lr)

        self.policy = GaussianPolicy(num_inputs,
                                     num_actions,
                                     q_net_hidden_size).to(self.device,
                                                           dtype=torch.float)
        self.policy_optim = Adam(self.policy.parameters(),
                                 lr=optimizer_lr)

    def select_action(self,
                      state,
                      evaluate=False):
        state = torch.from_numpy(state).to(self.device,
                                           dtype=torch.float)
        state = state.unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.as_tensor(state_batch).to(device=self.device,
                                                      dtype=torch.float)
        next_state_batch = torch.as_tensor(next_state_batch).to(device=self.device,
                                                                dtype=torch.float)
        action_batch = torch.as_tensor(action_batch).to(device=self.device,
                                                        dtype=torch.float)
        reward_batch = torch.as_tensor(reward_batch).to(self.device,
                                                        dtype=torch.float).unsqueeze(1)
        mask_batch = torch.as_tensor(mask_batch).to(self.device,
                                                    dtype=torch.float).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        state_batch.requires_grad = True
        action_batch.requires_grad = True
        next_q_value.requires_grad = True
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            # soft_update(self.critic_target, self.critic, self.tau)
            hard_update(self.critic_target, self.critic)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self,
                   env_name,
                   iteration,
                   save_model_dir,
                   actor_path=None,
                   critic_path=None,):
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        if actor_path is None:
            actor_path = os.path.join(save_model_dir,
                                      'actor_%s_%d' % (env_name, iteration))
        if critic_path is None:
            critic_path = os.path.join(save_model_dir,
                                       'critic_%s_%d' % (env_name, iteration))
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
