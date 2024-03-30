from collections import defaultdict
import csv

class TransitionRecorded:
    def __init__(self):
        self.state_counter = defaultdict(int) # Tracks frequency of each state
        self.transition_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Tracks transitions
        #A defaultdict where:
        #    - The first level keys are initial states,
        #    - The second level keys are actions taken from those states,
        #    - The third level keys are resulting states, with values being the count of occurrences.

        self.states = {} # stores state ID mapping
        
    def record_transition(self, current_state, next_state, action):
        self.state_counter[current_state] +=1
        self.transition_counter[current_state][action][next_state] += 1 #TODO: is this P(s'|s,a)?

    def calculate_probabilities(self):
        # Calculate state probabilities 
        total_states = sum(self.state_counter.values())
        state_probabilities = {state: count/total_states for state,count in self.state_counter.items()}
    
        #Calculate transition probabilities
        
        transition_probabilities = defaultdict(dict)
        # Iterate over each state and its corresponding actions in the transition counter.
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
    

    @staticmethod
    def process_and_save_transitions(trajectories, states_info):
        recorder = TransitionRecorded()

        for i in range(0, len(trajectories)-2, 2):
            current_state = trajectories[i]
            action = trajectories[i+1]
            next_state = trajectories[i+2]
            recorder.record_transition(current_state, next_state, action)

        recorder.save_to_csv(states_info)



    def save_to_csv(self, states_info):
        state_probabilities, transition_probabilities = self.calculate_probabilities()
        #Save states
        with open('states.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'value', 'p(s)', 'frequency'])
            for state_id, prob in state_probabilities.items():
                #Finds the state string corresponding to the given state ID in a more concise way.
                predicate = next((state_str for state_str, id in states_info.items() if id == state_id), "")
                writer.writerow([state_id, predicate, prob, self.state_counter[state_id]])

        #Save states
        with open('actions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['from', 'to', 'action','p(s)', 'frequency'])
            for (state, action, next_state), prob in transition_probabilities.items():
                writer.writerow([state, next_state, action, prob, self.transition_counter[state][action][next_state]])

