
from enum import Enum, auto
from av_discretizer import Action
from pgeon.agent import Agent

class SelfDrivingAgent(Agent):
    def __init__(self, environment ):
        super(SelfDrivingAgent, self).__init__()
        
        self.environment = environment
        self.policy = None #load PG.

        self.visited_states = set()
        self.newly_discovered_states = set()

    def act(self):
        # action = self.select_action(current_state)
        # new_state, reward, done, info = self.environment.step(action)
        # self.update_state_tracking(new_state)
        # Additional logic for learning from the transition (current_state, action, reward, new_state)
        current_state = self.environment.get_current_state()
        self.update_state_tracking(current_state)       
        pass

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

    
    #def update_policy(self, state, action, reward, next_state):
        # Here, the agent would update its policy based on the reward received and the transition
        # For our simple agent, there's no policy update logic
        #pass


# Example usage:
# env = SomeEnvironment()
# agent = Agent(env)

# Simulate agent's exploration process
# for _ in range(number_of_steps):
#     agent.act()

# At some point, compute the proportion of discovery
# print("Proportion of newly discovered states:", agent.compute_proportion_of_discovery())