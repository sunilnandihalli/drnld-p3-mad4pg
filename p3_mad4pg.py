import numpy as np
import random
import copy
from collections import namedtuple, deque
from p3_d4pg import D4PG, evaluating
from prioritized_memory import Memory
import torch
import torch.nn.functional as F

Vmax = 0.7
Vmin = -0.7
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExperienceQueue:
    def __init__(self, queue_length=100):
        self.states = deque(maxlen=queue_length)
        self.actions = deque(maxlen=queue_length)
        self.rewards = deque(maxlen=queue_length)


class MAD4PG:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size=int(1e6), batch_size=64, gamma=0.99, tau=1e-3,
                 update_every=3, num_mc_steps=5, num_agents=2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau
        self.UPDATE_EVERY = update_every
        self.num_mc_steps = num_mc_steps
        self.experiences = [ExperienceQueue(num_mc_steps) for _ in range(num_agents)]
        self.memory = Memory(buffer_size)
        self.t_step = 0
        self.train_start = batch_size
        self.mad4pg_agent = [D4PG(state_size, action_size, seed, device), D4PG(state_size, action_size, seed, device)]

    def acts(self, states, add_noise=0.0):
        acts = []
        for s, a in zip(states, self.mad4pg_agent):
            acts.append(a.act(np.expand_dims(s, 0), add_noise))
        return np.vstack(acts)

    # borrow from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter14    
    def distr_projection(self, next_distr_v, rewards_v, dones_mask_t, gamma):
        next_distr = next_distr_v.data.cpu().numpy()
        rewards = rewards_v.data.cpu().numpy()
        dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)
        dones_mask = np.squeeze(dones_mask)
        rewards = rewards.reshape(-1)

        for atom in range(N_ATOMS):
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
            b_j = (tz_j - Vmin) / DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l

            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l

            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
            b_j = (tz_j - Vmin) / DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            if dones_mask.shape == ():
                if dones_mask:
                    proj_distr[0, l] = 1.0
                else:
                    ne_mask = u != l
                    proj_distr[0, l] = (u - b_j)[ne_mask]
                    proj_distr[0, u] = (b_j - l)[ne_mask]
            else:
                eq_dones = dones_mask.copy()

                eq_dones[dones_mask] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l[eq_mask]] = 1.0
                ne_mask = u != l
                ne_dones = dones_mask.copy()
                ne_dones[dones_mask] = ne_mask
                if ne_dones.any():
                    proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                    proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr).to(device)

    def step(self, states, actions, rewards, next_states, dones):

        for agent_index in range(len(self.mad4pg_agent)):
            agent_experiences = self.experiences[agent_index]
            agent_experiences.states.appendleft(states[agent_index])
            agent_experiences.rewards.appendleft(rewards[agent_index] * self.GAMMA ** self.num_mc_steps)
            agent_experiences.actions.appendleft(actions[agent_index])
            for i in range(len(agent_experiences.rewards)):
                agent_experiences.rewards[i] /= self.GAMMA
            if len(agent_experiences.rewards) == self.num_mc_steps:  # N-steps return: r= r1+gamma*r2+..+gamma^(t-1)*rt
                state = torch.tensor(agent_experiences.states[-1]).float().unsqueeze(0).to(device)
                next_state = torch.tensor(next_states[agent_index]).float().unsqueeze(0).to(device)
                action = torch.tensor(agent_experiences.actions[-1]).float().unsqueeze(0).to(device)
                sum_reward = torch.tensor(sum(agent_experiences.rewards)).float().unsqueeze(0).to(device)
                with evaluating(self.mad4pg_agent[agent_index]) as cur_agent:
                    q_expected = cur_agent.critic_local(state, action)
                    action_next = cur_agent.actor_target(next_state)
                    q_target_next = cur_agent.critic_target(next_state, action_next)
                    q_target_next = F.softmax(q_target_next, dim=1)
                done_temp = torch.tensor(dones[agent_index]).float().to(device)
                q_target_next_projected = self.distr_projection(q_target_next, sum_reward, done_temp,
                                                                self.GAMMA ** self.num_mc_steps)
                q_target_next = -F.log_softmax(q_expected, dim=1) * q_target_next_projected
                error = q_target_next.sum(dim=1).mean().cpu().data
                self.memory.add(error, (states[agent_index], actions[agent_index], sum_reward,
                                        next_states[agent_index], dones[agent_index]))
            if dones[agent_index]:
                agent_experiences.states.clear()
                agent_experiences.rewards.clear()
                agent_experiences.actions.clear()

        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            # print(self.memory.tree.n_entries)
            if self.memory.tree.n_entries > self.train_start:
                common_experiences, idxs = self.sample()
                for agent_index in range(len(self.mad4pg_agent)):
                    self.learn(self.mad4pg_agent[agent_index], common_experiences, idxs)

    def sample(self):
        # prioritized experience replay
        mini_batch, idxs, is_weights = self.memory.sample(self.BATCH_SIZE)
        mini_batch = np.array(mini_batch).transpose()
        statess = np.vstack([m for m in mini_batch[0] if m is not None])
        actionss = np.vstack([m for m in mini_batch[1] if m is not None])
        rewardss = np.vstack([m for m in mini_batch[2] if m is not None])
        next_statess = np.vstack([m for m in mini_batch[3] if m is not None])
        doness = np.vstack([m for m in mini_batch[4] if m is not None])
        # bool to binary
        doness = doness.astype(int)
        statess = torch.from_numpy(statess).float().to(device)
        actionss = torch.from_numpy(actionss).float().to(device)
        rewardss = torch.from_numpy(rewardss).float().to(device)
        next_statess = torch.from_numpy(next_statess).float().to(device)
        doness = torch.from_numpy(doness).float().to(device)
        return (statess, actionss, rewardss, next_statess, doness), idxs

    def learn(self, agent, experiences, idxs):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Compute critic loss
        q_expected = agent.critic_local(states, actions)
        actions_next = agent.actor_target(next_states)
        q_targets_next = agent.critic_target(next_states, actions_next)
        q_targets_next = F.softmax(q_targets_next, dim=1)
        q_targets_next = self.distr_projection(q_targets_next, rewards, dones, self.GAMMA ** self.num_mc_steps)
        q_targets_next = -F.log_softmax(q_expected, dim=1) * q_targets_next
        critic_loss = q_targets_next.sum(dim=1).mean()
        with torch.no_grad():
            errors = q_targets_next.sum(dim=1).cpu().data.numpy()
        # update priority
        for i in range(self.BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Compute actor loss
        actions_pred = agent.actor_local(states)
        crt_distr_v = agent.critic_local(states, actions_pred)
        actor_loss = -agent.critic_local.distr_to_q(crt_distr_v)
        actor_loss = actor_loss.mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, self.TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, self.TAU)
