from pgeon.policy_graph import PolicyGraph
from math import log2
from collections import defaultdict

class PolicyGraphEvaluator:
    def __init__(self, policy_graph: PolicyGraph):
        self.policy_graph = policy_graph




    def compute_entropy_metrics(self):
        """
        Compute the entropy metrics for the Policy Graph.

        Returns:
        - A dictionary containing the values of H(s), Ha(s), and Hw(s) for each state.
        """
        entropy_metrics = {}
        
        for state in self.policy_graph.nodes():
            # Compute probabilities for actions given the current state
            action_freq = defaultdict(int)
            total_action_freq = 0
            
            for _, next_state, data in self.policy_graph.out_edges(state, data=True):
                action = data['action']
                freq = data['frequency']
                action_freq[action] += freq
                total_action_freq += freq
            
            Ha = sum(-(freq / total_action_freq) * log2(freq / total_action_freq) for freq in action_freq.values())

            Hw = 0
            for action, freq in action_freq.items():
                P_a_given_s = freq / total_action_freq
                action_specific_out_edges = filter(lambda edge: edge[2]['action'] == action, self.policy_graph.out_edges(state, data=True))
                for _, next_state, data in action_specific_out_edges:
                    prob = data['probability']
                    Hw -= P_a_given_s * prob * log2(prob)

            Hs = Ha + Hw

            entropy_metrics[state] = {'Hs': Hs, 'Ha': Ha, 'Hw': Hw}
        
        return entropy_metrics
        
    def compute_scene_reward(self, trajectories):
        #TODO: handle discretize state before computing reward,
        #TODO: fix
        ''' Given an episode/scene with trajectory [state, action, next_state, ...], computes the average reward.'''

        tot_reward = 0
        for i in range(0, len(trajectories)-2, 2):
            current_state = trajectories[i]
            action = trajectories[i+1]
            next_state = trajectories[i+2]
            tot_reward += self.policy_graph.compute_reward(current_state, action, next_state)
        return tot_reward

    def calculate_tl(self):
        """Calculates the Transferred Learning (TL) metric."""
        pass

    def calculate_std(self):
        """Calculates the Standard Deviation (STD) metric."""
        pass

    def calculate_ns(self):
        """Calculates the New States (NS) visited metric."""
        pass        
