from pgeon.policy_graph import PolicyGraph
from math import log2
from collections import defaultdict
import numpy as np


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
           
    '''
    def update_state_tracking(self, state):
        """
        Updates the visited and newly discovered states based on the current state.
        """
        if state not in self.visited_states:
            self.newly_discovered_states.add(state)
        self.visited_states.add(state)


    def compute_proportion_of_discovery(self):
        """
        Computes the proportion of newly discovered states to total visited states.
        """
        if len(self.visited_states) == 0:
            return 0  # Prevent division by zero
        return len(self.newly_discovered_states) / len(self.visited_states)
    '''

