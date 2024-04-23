from collections import defaultdict
import csv
import pandas as pd
import os
from pgeon.policy_graph import PolicyGraph

class TransitionRecorded:
    def __init__(self):
        self.state_counter = defaultdict(int) # Tracks frequency of each state
        self.transition_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # Tracks transitions with counts.
        # Structure: {state: {action: {next_state: count}}}
        self.destination_states = set()

    def record_transition(self, current_state, next_state, action):
        self.state_counter[current_state] +=1
        if action is not None:
            self.transition_counter[current_state][action][next_state] += 1 
        else:
            self.destination_states.add(current_state)

    def calculate_probabilities(self):
        total_states = sum(self.state_counter.values())
        state_probabilities = {state: count/total_states for state,count in self.state_counter.items()}
            
        transition_probabilities = defaultdict(dict)
        
        for state, actions in self.transition_counter.items():
            # For each action, iterate over the next states and their counts.
            for action, transitions in actions.items():
                # Calculate the total number of transitions for the current action from the current state
                total_transitions = sum(transitions.values())
                
                # For each next state following the current action, calculate the transition probability.
                # This is done by dividing the count of the specific transition by the total transitions for the action.
                for next_state, count in transitions.items():
                    transition_probabilities[(state,action,next_state)] = count/ total_transitions

        return state_probabilities, transition_probabilities
    

    def process_transitions(self, trajectories):#, states_info, path = '.'):
        for i in range(0, len(trajectories)-2, 2):
            current_state = trajectories[i]
            action = trajectories[i+1]
            next_state = trajectories[i+2]
            self.record_transition(current_state, next_state, action)



    def save_to_csv(self, states_info, path="."):
        state_probabilities, transition_probabilities = self.calculate_probabilities()
        states_file_path = os.path.join(path, 'nuscenes_nodes.csv')
        with open(states_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'value', 'p(s)', 'frequency','is_destination'])
            for state_id, prob in state_probabilities.items():
                #Finds the state string corresponding to the given state ID in a more concise way.
                predicate = next((state_str for state_str, id in states_info.items() if id == state_id), "")
                if state_id in self.destination_states:
                    writer.writerow([state_id, predicate, prob, self.state_counter[state_id], 1])
                else:
                    writer.writerow([state_id, predicate, prob, self.state_counter[state_id], 0])
                


        actions_file_path = os.path.join(path, 'nuscenes_edges.csv')
        with open(actions_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['from', 'to', 'action','p(s)', 'frequency'])
            for (state, action, next_state), prob in transition_probabilities.items():
                writer.writerow([state, next_state, action, prob, self.transition_counter[state][action][next_state]])



   
    def create_dataframes(self, states_info):
        """
        Function that creates dataframe of states and actions of the agent.
        """
        state_probabilities, transition_probabilities = self.calculate_probabilities()
        
        states_data = []
        for state_id, prob in state_probabilities.items():
            predicate = next((state_str for state_str, id in states_info.items() if id == state_id), "")
            is_destination = 1 if state_id in self.destination_states else 0
            states_data.append([state_id, predicate, prob, self.state_counter[state_id], is_destination])
        
        states_df = pd.DataFrame(states_data, columns=['id', 'value', 'p(s)', 'frequency', 'is_destination'])
        
        actions_data = []
        for (state, action, next_state), prob in transition_probabilities.items():
            actions_data.append([state, next_state, action, prob, self.transition_counter[state][action][next_state]])
        
        actions_df = pd.DataFrame(actions_data, columns=['from', 'to', 'action', 'p(s)', 'frequency'])
        
        return states_df, actions_df
