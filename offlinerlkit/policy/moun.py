import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import BaseDynamics,EnsembleDynamics

from scipy.spatial import KDTree

class MOUNPolicy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    mopo_target_Q

    """

    def __init__(
            self,
            dynamics: BaseDynamics,
            actor: nn.Module,
            critic1: nn.Module,
            critic2: nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1_optim: torch.optim.Optimizer,
            critic2_optim: torch.optim.Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics

    def rollout(
            self,
            init_obss: np.ndarray,
            rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals = self.dynamics.step_no_penalty(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
               {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
                                                       mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]


        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)

            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs

            target_q = rewards + self._gamma * (1 - terminals) * next_q
        next_actions_np = actions.cpu()
        next_obss_np = obss.cpu()
        penalty = self.dynamics.get_uncertainty(next_obss_np, next_actions_np)
        penalty = torch.tensor(penalty)
        penalty = penalty.cuda()
        vt = q1 - target_q
        vt1 = vt * penalty

        critic1_loss = (vt1.pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        vt = q2 - target_q
        vt2 = vt * penalty

        critic2_loss = (vt2.pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()


        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

class MOUN1Policy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    mix - CQL_next_Q  - penalty_reward
    """

    def __init__(
            self,
            dynamics: BaseDynamics,
            actor: nn.Module,
            critic1: nn.Module,
            critic2: nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1_optim: torch.optim.Optimizer,
            critic2_optim: torch.optim.Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics

    def rollout(
            self,
            init_obss: np.ndarray,
            rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, infos = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
               {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        obss, actions, next_obss, rewards, terminals = real_batch["observations"], real_batch["actions"], \
                                                       real_batch["next_observations"], real_batch["rewards"], real_batch["terminals"]

        fake_obss, fake_actions, fake_next_obss, fake_rewards, fake_terminals = fake_batch["observations"], fake_batch["actions"], \
                                                       fake_batch["next_observations"], fake_batch["rewards"], \
                                                       fake_batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_actions_np = actions.cpu()
            next_obss_np = obss.cpu()
            penalty = self.dynamics.get_uncertainty(next_obss_np, next_actions_np)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = (rewards.cpu() - penalty).cuda() + self._gamma * (1 - terminals) * next_q
            vt = (q1 - target_q).cpu()
            vt = [x*y  for x,y in zip(vt,penalty)]
            vt = (torch.tensor(vt)).cuda()
        critic1_loss = (vt.pow(2)).mean()
        # self.critic1_optim.zero_grad()
        # critic1_loss.backward()
        # self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        # self.critic2_optim.zero_grad()
        # critic2_loss.backward()
        # self.critic2_optim.step()

        with torch.no_grad():
            fake_next_actions, fake_next_log_probs = self.actforward(fake_next_obss)
            fake_next_q = torch.min(
                self.critic1_old(fake_next_obss, fake_next_actions), self.critic2_old(fake_next_obss, fake_next_actions)
            ) - self._alpha * fake_next_log_probs
            fake_target_q = fake_rewards + self._gamma * (1 - fake_terminals) * fake_next_q

        fake_q1, fake_q2 = self.critic1(fake_obss, fake_actions), self.critic2(fake_obss, fake_actions)
        fake_critic1_loss = ((fake_q1 - fake_target_q).pow(2)).mean()
        critic1_loss = critic1_loss + fake_critic1_loss
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        fake_critic2_loss = ((fake_q2 - fake_target_q).pow(2)).mean()
        critic2_loss = critic2_loss + fake_critic2_loss
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

class MOUN2Policy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    mix - cql - penalty_next_Q
    """

    def __init__(
            self,
            dynamics: BaseDynamics,
            actor: nn.Module,
            critic1: nn.Module,
            critic2: nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1_optim: torch.optim.Optimizer,
            critic2_optim: torch.optim.Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics

    def rollout(
            self,
            init_obss: np.ndarray,
            rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals = self.dynamics.step_no_penalty(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
               {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        obss, actions, next_obss, rewards, terminals = real_batch["observations"], real_batch["actions"], \
                                                       real_batch["next_observations"], real_batch["rewards"], real_batch["terminals"]

        fake_obss, fake_actions, fake_next_obss, fake_rewards, fake_terminals = fake_batch["observations"], fake_batch["actions"], \
                                                       fake_batch["next_observations"], fake_batch["rewards"], \
                                                       fake_batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(fake_next_obss)
            next_q = torch.min(
                self.critic1_old(fake_next_obss, next_actions), self.critic2_old(fake_next_obss, next_actions)
            ) - self._alpha * next_log_probs
            next_actions_np = next_actions.cpu()
            next_obss_np = fake_next_obss.cpu()
            penalty = self.dynamics.get_uncertainty(next_obss_np, next_actions_np)
            next_q = next_q.cpu() - penalty
            target_q = fake_rewards + self._gamma * (1 - fake_terminals) * next_q.cuda()

        q1, q2 = self.critic1(fake_obss, fake_actions), self.critic2(fake_obss, fake_actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

class MOUN3Policy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    mix - cql - penalty_next_Q - jia quan
    """

    def __init__(
            self,
            data,
            state_dim,
            action_dim,
            penalty_coef,
            prdc,
            dynamics: BaseDynamics,
            actor: nn.Module,
            critic1: nn.Module,
            critic2: nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1_optim: torch.optim.Optimizer,
            critic2_optim: torch.optim.Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
            num_samples: int = 10,
            beta = 2,  # [beta* state, action]
            k = 1,
            device: str = "cuda"
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics
        self._num_samples = num_samples
        self.k = k
        # KD-Tree
        self.beta = beta
        self.data = data
        self.kd_tree = KDTree(data)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.penalty_coef = penalty_coef
        self.prdc = prdc

    def rollout(
            self,
            init_obss: np.ndarray,
            rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals = self.dynamics.step_no_penalty(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
               {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
    
    @ torch.no_grad()
    def compute_lcb(self, obss: torch.Tensor, actions: torch.Tensor):
        # compute next q std
        pred_next_obss = self.dynamics.sample_next_obss(obss, actions, self._num_samples)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
        pred_next_actions, _ = self.actforward(pred_next_obss)
        
        pred_next_qs =  torch.cat([critic_old(pred_next_obss, pred_next_actions) for critic_old in [self.critic1_old,self.critic2_old]], 1)
        pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
        penalty = pred_next_qs.mean(0).std(0)

        return penalty

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        obss, actions, next_obss, rewards, terminals = real_batch["observations"], real_batch["actions"], \
                                                       real_batch["next_observations"], real_batch["rewards"], real_batch["terminals"]

        fake_obss, fake_actions, fake_next_obss, fake_rewards, fake_terminals = fake_batch["observations"], fake_batch["actions"], \
                                                       fake_batch["next_observations"], fake_batch["rewards"], \
                                                       fake_batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        with torch.no_grad():
            penalty = self.compute_lcb(fake_next_obss, fake_actions)
            next_actions, next_log_probs = self.actforward(fake_next_obss)
            next_q = torch.min(
                self.critic1_old(fake_next_obss, next_actions), self.critic2_old(fake_next_obss, next_actions)
            ) - self._alpha * next_log_probs

            ## Get the nearest neighbor
            key = torch.cat([self.beta * fake_obss, fake_actions], dim=1).detach().cpu().numpy()
            _, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
            ## Calculate the regularization
            nearest_neightbour = (
                torch.tensor(self.data[idx][:, :, -self.action_dim:])
                    .squeeze(dim=1)
                    .to(self.device)
            )
            data_c = np.abs(fake_actions.cpu() - nearest_neightbour.cpu())
            dc_loss = data_c.mean(dim=1)
            dc1_loss = dc_loss.view(-1, 1).cuda()

            target_q = (fake_rewards - self.penalty_coef * penalty - self.prdc * dc1_loss) + self._gamma * (1 - fake_terminals) * next_q.cuda()

        q1, q2 = self.critic1(fake_obss, fake_actions), self.critic2(fake_obss, fake_actions)
        fake_critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic1_loss = critic1_loss * 0.2 + fake_critic1_loss * 0.8
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        fake_critic2_loss = ((q2 - target_q).pow(2)).mean()
        critic2_loss = critic2_loss * 0.2 + fake_critic2_loss + 0.8
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result


class MOUN4Policy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    mix - cql - penalty_next_Q - jia quan - policy-penalty
    """

    def __init__(
            self,
            data,
            state_dim,
            action_dim,
            penalty_coef,
            prdc,
            dynamics: BaseDynamics,
            actor: nn.Module,
            critic1: nn.Module,
            critic2: nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1_optim: torch.optim.Optimizer,
            critic2_optim: torch.optim.Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
            beta=2,  # [beta* state, action]
            k=1,
            device: str = "cuda",
            for_num: int = 1
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics

        self.k = k
        # KD-Tree
        self.beta = beta
        self.data = data
        self.kd_tree = KDTree(data)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.penalty_coef = penalty_coef
        self.prdc = prdc
        self.forNum = for_num

    def rollout(
            self,
            init_obss: np.ndarray,
            rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            # actions = self.select_action(observations)
            action, _ = self.actforward(observations, False)
            actions = action.detach().cpu().numpy()
            next_observations, rewards, terminals = self.dynamics.step_no_penalty(observations, actions)

            next_actions_np = actions
            next_obss_np = observations
            penalty = self.dynamics.get_uncertainty(next_obss_np, next_actions_np)

            next_obss_np = torch.tensor(next_obss_np)
            next_actions_np = torch.tensor(next_actions_np)
            
            ## Get the nearest neighbor
            key = torch.cat([self.beta * next_obss_np, next_actions_np], dim=1).detach().cpu().numpy()
            _, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
            ## Calculate the regularization
            nearest_neightbour = (
                torch.tensor(self.data[idx][:, :, -self.action_dim:])
                    .squeeze(dim=1)
                    .to(self.device)
            )
            # prdc = torch.abs(action - nearest_neightbour)
            # mask_prdc = prdc < 0.2
            # prdc[mask_prdc] = 0.2
            # mask_prdc = prdc >= 0.2
            # prdc[mask_prdc] = -prdc[mask_prdc]
            # dc_loss = F.mse_loss(action, nearest_neightbour)
            data_c = np.abs(next_actions_np - nearest_neightbour.cpu())
            dc_loss = data_c.mean(dim=1)
            dc1_loss = dc_loss.view(-1,1).numpy()
            reward = rewards - self.penalty_coef * penalty - self.prdc * dc1_loss

            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(reward)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
               {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def get_q(self, obs, act):
        return self.critic1(obs, act)

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obssS, actionsS, next_obssS, rewardsS, terminalsS = mix_batch["observations"], mix_batch["actions"], \
                                                       mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]

        for i in range(self.forNum):

            q1, q2 = self.critic1(obssS, actionsS), self.critic2(obssS, actionsS)
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obssS)
                next_q = torch.min(
                    self.critic1_old(next_obssS, next_actions), self.critic2_old(next_obssS, next_actions)
                ) - self._alpha * next_log_probs

                target_q = rewardsS + self._gamma * (1 - terminalsS) * next_q  #+ self.prdc * dc1_loss

            critic1_loss = ((q1 - target_q).pow(2)).mean()
            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            self.critic1_optim.step()

            critic2_loss = ((q2 - target_q).pow(2)).mean()
            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obssS)
        q1a, q2a = self.critic1(obssS, a), self.critic2(obssS, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()# + dc_loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result