from collections import defaultdict
from enum import Enum, auto
from typing import Tuple, Any, List, Union, Set
import csv
import pickle
import gymnasium as gym
import networkx as nx
import numpy as np
import tqdm
from pgeon.agent import Agent
from pgeon.discretizer import Predicate, Discretizer
import time
import pandas as pd
from example.discretizer.utils import Action
import math

class PolicyGraph(nx.MultiDiGraph):

    ######################
    # CREATION/LOADING
    ######################

    def __init__(self,
                 environment: gym.Env,
                 discretizer: Discretizer
                 ):
        super().__init__()
        self.environment = environment
        self.discretizer = discretizer

        #self.unique_states: Dict[str, int] = {}
        self.state_to_be_discretized = ['x', 'y', 'velocity', 'steering_angle'] 
        #these columns are used only to determine the action between 2 states in the training scenes 
        self.state_columns_for_action = ['delta_local_x', 'delta_local_y', 'velocity', 'acceleration', 'steering_angle'] #heading_change_rate not needed
     
        # Metrics of the teacher
        self.agent_metrics = {'AER': [], 'STD': []}
        

        self._is_fit = False
        self._trajectories_of_last_fit: List[List[Any]] = []



    @staticmethod
    def from_pickle(path: str):
        path_includes_pickle = path[-7:] == '.pickle'
        with open(f'{path}{"" if path_includes_pickle else ".pickle"}', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def from_data_frame(states_df, actions_df, environment, discretizer):
        pg = PolicyGraph(environment, discretizer)
        node_info = {}
        
        for index, row in states_df.iterrows():
            state_id = int(row['id'])
            value = row['value']
            state_prob = float(row['p(s)'])
            state_freq = int(row['frequency'])
            is_destination = row['is_destination']
            #is_destination = True if row['is_destination'] == 1 else False            
            node_info[state_id] = {
                'value': pg.discretizer.str_to_state(value),
                'probability': state_prob,
                'frequency': state_freq,
                'is_destination': is_destination
            }
            pg.add_node(node_info[state_id]['value'], 
                    probability=state_prob,
                    frequency=state_freq,
                    is_destination=is_destination,
                    )
            
        for index, row in actions_df.iterrows():
            state_from = int(row['from'])
            state_to = int(row['to'])
            action = int(row['action'])
            prob = float(row['p(s)'])
            freq = int(row['frequency'])
            #is_destination = True if node_info[state_to]['is_destination'] == '1' else False
            
            pg.add_edge(node_info[state_from]['value'], node_info[state_to]['value'], key=action,
                        frequency=freq, probability=prob, action=action)#, reward=environment.compute_reward(node_info[state_from]['value'], action, node_info[state_to]['value'], is_destination))

        pg._is_fit = True
        return pg

    
    @staticmethod
    def from_nodes_and_edges(path_nodes: str,
                             path_edges: str,
                             environment: gym.Env,
                             discretizer: Discretizer):
        pg = PolicyGraph(environment, discretizer)

        path_to_nodes_includes_csv = path_nodes[-4:] == '.csv'
        path_to_edges_includes_csv = path_edges[-4:] == '.csv'

        node_info = {}
        with open(f'{path_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for state_id, value, prob, freq, is_destination in csv_r:
                state_prob = float(prob)
                state_freq = int(freq)

                node_info[int(state_id)] = {
                    'value': pg.discretizer.str_to_state(value),
                    'probability': state_prob,
                    'frequency': state_freq,
                    'is_destination': is_destination
                }
                
                pg.add_node(node_info[int(state_id)]['value'], 
                            probability=state_prob,
                            frequency=state_freq,
                            is_destination=is_destination)

        with open(f'{path_edges}{"" if path_to_edges_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for node_from, node_to, action, prob, freq in csv_r:
                node_from = int(node_from)
                node_to = int(node_to)
                # TODO Get discretizer to process the action id correctly;
                #  we cannot assume the action will always be an int
                action = int(action)
                prob = float(prob)
                freq = int(freq)
                is_destination = True if node_info[node_to]['is_destination'] == '1' else False
                pg.add_edge(node_info[node_from]['value'], node_info[node_to]['value'], key=action,
                            frequency=freq, probability=prob, action=action)#, reward=environment.compute_reward(node_info[node_from]['value'], action, node_info[node_to]['value'], is_destination))
        pg._is_fit = True
        return pg


    @staticmethod
    def from_nodes_and_trajectories(path_nodes: str,
                                    path_trajectories: str,
                                    environment: gym.Env,
                                    discretizer: Discretizer):
        pg = PolicyGraph(environment, discretizer)

        path_to_nodes_includes_csv = path_nodes[-4:] == '.csv'
        path_to_trajs_includes_csv = path_trajectories[-4:] == '.csv'

        node_info = {}
        with open(f'{path_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for state_id, value, prob, freq in csv_r:
                state_prob = float(prob)
                state_freq = int(freq)

                node_info[int(state_id)] = {
                    'value': pg.discretizer.str_to_state(value),
                    'probability': state_prob,
                    'frequency': state_freq
                }

        with open(f'{path_trajectories}{"" if path_to_trajs_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)

            for csv_trajectory in csv_r:
                trajectory = []
                for elem_position, element in enumerate(csv_trajectory):
                    # Process state
                    if elem_position % 2 == 0:
                        trajectory.append(node_info[int(element)]['value'])
                    # Process action
                    else:
                        trajectory.append(int(element))

                pg._update_with_trajectory(trajectory)
                pg._trajectories_of_last_fit.append(trajectory)

        pg._is_fit = True
        return pg

    ######################
    # FITTING
    ######################

    def _run_episode(self,
                     scene,
                     #max_steps: int = 100,
                     #seed: int = None,
                     verbose = False
                     ) -> List[Any]:

        """
            Discretizes a trajectory (list of states) and stores unique states and actions.

            Args:
                states: DataFrame containing state information of a scene for each time step.

            Returns:
                List containing tuples of (current state ID, action ID, next state ID).
        """
        self.detection_cameras = [col for col in scene.columns if 'CAM' in col] 
        #NOTE: this should be assigned even when testing the PG (not only when computing the trajectory)
        
        trajectory = []
        tot_reward = 0
        for i in range(len(scene)-1):
            
            # discretize current state
            current_state_to_discretize = scene.iloc[i][self.state_to_be_discretized].tolist()
            current_detection_info = scene.iloc[i][self.detection_cameras] if self.detection_cameras else None
            discretized_current_state = self.discretizer.discretize(current_state_to_discretize, current_detection_info)
            #current_state_id = self.add_unique_state(current_state_str)


            current_state_for_action = scene.iloc[i][self.state_columns_for_action].tolist()
            next_state_for_action = scene.iloc[i+1][self.state_columns_for_action].tolist()
            action = self.discretizer.determine_action(current_state_for_action, next_state_for_action)
            action_id = self.discretizer.get_action_id(action)
   
            trajectory.extend([discretized_current_state, action_id])        

            #TODO: also consider objects in the reward. Do i need to add reward to last state?
            reward = self.environment.compute_reward(discretized_current_state, action)
            tot_reward +=sum(reward)

            if verbose:
                print('From', current_state_to_discretize, ' -> ', action)
                print(f'Rewards: {reward}')

        #add last state
        last_state_to_discretize = scene.iloc[len(scene)-1][self.state_to_be_discretized].tolist()
        last_state_detections = scene.iloc[len(scene)-1][self.detection_cameras] if self.detection_cameras else None
        discretized_last_state = self.discretizer.discretize(last_state_to_discretize, last_state_detections)
        #last_state_id = self.add_unique_state(last_state_str)

        trajectory.append(discretized_last_state)
        self.environment.stop_points.add(discretized_last_state)     
        reward = self.environment.compute_reward(discretized_last_state, None)
        tot_reward +=sum(reward)

        if verbose:
                print('From', last_state_to_discretize, ' -> END ')
                print(f'Rewards: {reward}')

        return trajectory, tot_reward
    
    
    

    def _update_with_trajectory(self,
                                trajectory: List[Any]
                                ):

        # Only even numbers are states
        states_in_trajectory = [trajectory[i] for i in range(len(trajectory)) if i % 2 == 0]
        all_new_states_in_trajectory = {state for state in set(states_in_trajectory) if not self.has_node(state)}
        self.add_nodes_from(all_new_states_in_trajectory, frequency=0, is_destination=0)

        state_frequencies = {s: states_in_trajectory.count(s) for s in set(states_in_trajectory)}
        for state in state_frequencies:
            self.nodes[state]['frequency'] += state_frequencies[state]

        pointer = 0
        while (pointer + 1) < len(trajectory):
            state_from, action, state_to = trajectory[pointer:pointer + 3]
            if not self.has_edge(state_from, state_to, key=action):
                self.add_edge(state_from, state_to, key=action, frequency=0, action=action)
            self[state_from][state_to][action]['frequency'] += 1
            pointer += 2

        last_state = states_in_trajectory[-1]
        if last_state in self.nodes:
            self.nodes[last_state]['is_destination'] = 1


    def _normalize(self):
        weights = nx.get_node_attributes(self, 'frequency')
        total_frequency = sum([weights[state] for state in weights]) #total number of edges in pg
        nx.set_node_attributes(self, {state: weights[state] / total_frequency for state in weights}, 'probability') #coincide con p(s)

        for node in self.nodes:
            total_frequency = sum([self.get_edge_data(node, dest_node, action)['frequency']
                                   for dest_node in self[node]
                                   for action in self.get_edge_data(node, dest_node)])
            for dest_node in self[node]:
                for action in self.get_edge_data(node, dest_node):
                    self[node][dest_node][action]['probability'] = \
                        self[node][dest_node][action]['frequency'] / total_frequency


    def fit(self,
            scenes: pd.DataFrame,
            update: bool = False,
            verbose = True
            ):

        if not update:
            self.clear()
            self._trajectories_of_last_fit = []
            self._is_fit = False

        scene_groups = scenes.groupby('scene_token')
        progress_bar = tqdm.tqdm(total=len(scene_groups), desc='Fitting PG from scenes...')

        progress_bar.set_description('Fitting PG from scenes...')

        start_time = time.time()
        rewards = []


        for scene_token, group in scene_groups:
            trajectory_result, tot_reward = self._run_episode(group, verbose)
            self._update_with_trajectory(trajectory_result)
            self._trajectories_of_last_fit.append(trajectory_result)

            rewards.append(tot_reward)
            progress_bar.update(1)

            if verbose:
                print('Final scene reward: ', tot_reward)

        self._normalize()
        self._is_fit = True


        # Compute the average reward and std
        average_reward = sum(rewards) / scene_groups.ngroups
        std = np.std(rewards)
        self.agent_metrics['AER'].append(average_reward)
        self.agent_metrics['STD'].append(std)
        self.epoch_mean_time = time.time() - start_time
        # Compute how much time we spent
        print(f"Average Reward: {average_reward} and Standard Deviation: {std} --> Epoch Mean Time: {self.epoch_mean_time}")

        return self
    
    
    '''
    def _run_episode(self,
                     agent: Agent,
                     max_steps: int = 100,
                     seed: int = None,
                     verbose = False
                     ) -> List[Any]:

        """
        #TODO: Suppose the starts from a random point and does not have a destination. 
        Check and evaluate the behavior untill the end of the episode.
        You could compare the trajectory of the real agent (nuscene scenario) and this trajectory.
        """ 
        observation = self.environment.reset(seed=seed)
        done = False
        trajectory = [self.discretizer.discretize(observation)]
        if verbose:
                print('Initial State:', observation)
        step_counter = 0
        tot_reward = 0
        while not done:
            if step_counter >= max_steps:
                break
            
            action, is_destination = agent.act(observation)
            observation, reward, _, _ = self.environment.step(action, is_destination)
            trajectory.extend([action, self.discretizer.discretize(observation)])

            tot_reward +=reward
            step_counter += 1

            if verbose:
                    print('Action:', action)
                    print('Reward:', reward)
                    print('Map in Next state:\n')
                    print(observation)


        return trajectory, tot_reward
    '''
    
    
    
    '''
    def compute_total_reward(self, agent, scenes, max_steps=100):
        """
        Computes the total reward obtained by following a policy from an initial state to a final state.
        
        Args:
        - agent: The agent following the policy.
        - initial_state: The state from which to start (discretized).
        - final_state: The final state to reach (discretized)
        - max_steps: Maximum number of steps to prevent infinite loops.
        
        Returns:
        - total_reward: The total reward obtained.
        - reached_final: Boolean indicating if the final state was reached.
        """
        total_reward = 0
        step_count = 0
        reached_final = False

        self.current_state = initial_state

        while step_count < max_steps:
            action_id, is_destination = agent.act(self.current_state)
            action = self.discretizer.get_action_from_id(action_id)
            next_state, reward, _, _ = self.step(action, is_destination)
            total_reward +=reward
            step_count +=1

            print(f'step count: {step_count}')
            self.current_state = next_state
            
            if next_state == final_state: #TODO: fix
                reached_final = True
                break

        return total_reward, reached_final
        
        '''
    



    ######################
    # EXPLANATIONS
    ######################
    def get_nearest_predicate(self, input_predicate: Tuple[Enum], verbose=False):
        """ Returns the nearest predicate on the PG. If already exists, then we return the same predicate. If not,
        then tries to change the predicate to find a similar state (Maximum change: 1 value).
        If we don't find a similar state, then we return None

        :param input_predicate: Existent or non-existent predicate in the PG
        :return: Nearest predicate
        :param verbose: Prints additional information
        """
        # Predicate exists in the MDP
        if self.has_node(input_predicate):
            if verbose:
                print('NEAREST PREDICATE of existing predicate:', input_predicate)
            return input_predicate
        else:
            if verbose:
                print('NEAREST PREDICATE of NON existing predicate:', input_predicate)

            nearest_state_generator = self.discretizer.nearest_state(input_predicate)
            new_predicate = input_predicate
            try:
                while not self._is_predicate_in_pg_and_usable(new_predicate):
                    new_predicate = next(nearest_state_generator)

            except StopIteration:
                print("No nearest states available.")
                new_predicate = None

            if verbose:
                print('\tNEAREST PREDICATE in PG:', new_predicate)
            return new_predicate     
        
    




    def get_possible_actions(self, predicate):
        """ Given a predicate, get the possible actions and it's probabilities

        3 cases:

        - Predicate not in PG but similar predicate found in PG: Return actions of the similar predicate
        - Predicate not in PG and no similar predicate found in PG: Return all actions same probability
        - Predicate in MDP: Return actions of the predicate in PG

        :param predicate: Existing or not existing predicate
        :return: Action probabilities of a given state
        """
        result = defaultdict(float)

        # Predicate not in PG
        if predicate not in self.nodes():
            # Nearest predicate not found -> Random action
            if predicate is None:
                result = {action: 1 / len(self.discretizer.all_actions()) for action in self.discretizer.all_actions()}
                return sorted(result.items(), key=lambda x: x[1], reverse=True)
            predicate = self.get_nearest_predicate(predicate)
            if predicate is None:
                result = {a: 1 / len(self.discretizer.all_actions()) for a in self.discretizer.all_actions()}
                return list(result.items())
        # Out edges with actions [(u, v, a), ...]
        possible_actions = [(u, data['action'], v, data['probability'])
                            for u, v, data in self.out_edges(predicate, data=True)]
        """
        for node in self.pg.nodes():
            possible_actions = [(u, data['action'], v, data['weight'])
                                for u, v, data in self.pg.out_edges(node, data=True)]
            s = sum([w for _,_,_,w in possible_actions])
            assert  s < 1.001 and s > 0.99, f'Error {s}'
        """
        # Drop duplicated edges
        possible_actions = list(set(possible_actions))
        # Predicate has at least 1 out edge.
        if len(possible_actions) > 0:
            for _, action, v, weight in possible_actions:
                #NOTE: I subtract -1 since the Actions have id starting from 1 (it's an Enum).
                #otherwise i get index out of bound error
                result[self.discretizer.all_actions()[action-1]] += weight
            return sorted(result.items(), key=lambda x: x[1], reverse=True)
        # Predicate does not have out edges. Then return all the actions with same probability
        else:
            result = {a: 1 / len(self.discretizer.all_actions()) for a in self.discretizer.all_actions()}
            return list(result.items())

    def question1(self, predicate, verbose=False):
        possible_actions = self.get_possible_actions(predicate)
        if verbose:
            print('I will take one of these actions:')
            for action, prob in possible_actions:
                print('\t->', action.name, '\tProb:', round(prob * 100, 2), '%')
        return possible_actions

    def get_when_perform_action(self, action):
        """ When do you perform action

        :param action: Valid action
        :return: Set of states that has an out edge with the given action
        """
        # Nodes where 'action' it's a possible action
        # All the nodes that has the same action (It has repeated nodes)
        all_nodes = [u for u, v, a in self.edges(data='action') if a == action]
        # Drop all the repeated nodes
        all_nodes = list(set(all_nodes))

        # Nodes where 'action' it's the most probable action
        all_edges = [list(self.out_edges(u, data=True)) for u in all_nodes]

        all_best_actions = [
            sorted([(u, data['action'], data['probability']) for u, v, data in edges], key=lambda x: x[2], reverse=True)[0]
            for edges in all_edges]
        best_nodes = [u for u, a, w in all_best_actions if a == action]

        all_nodes.sort()
        best_nodes.sort()
        return all_nodes, best_nodes

    def question2(self, action, verbose=False):
        """
        Answers the question: When do you perform action X?
        """
        if verbose:
            print('*********************************')
            print(f'* When do you perform action {action}?')
            print('*********************************')

        all_nodes, best_nodes = self.get_when_perform_action(action)
        if verbose:
            print(f"Most probable in {len(best_nodes)} states:")
        for i in range(len(all_nodes)):
            if i < len(best_nodes) and verbose:
                print(f"\t-> {best_nodes[i]}")
        # TODO: Extract common factors of resulting states
        return best_nodes

    def get_most_probable_option(self, predicate, greedy=False, verbose=False):
        if greedy:
            nearest_predicate = self.get_nearest_predicate(predicate, verbose=verbose)
            possible_actions = self.get_possible_actions(nearest_predicate)

            # Possible actions always will have 1 element since  for each state we only save the best action
            return possible_actions[0][0]
        else:
            nearest_predicate = self.get_nearest_predicate(predicate, verbose=verbose)
            possible_actions = self.get_possible_actions(nearest_predicate)
            possible_actions = sorted(possible_actions, key=lambda x: x[1])
            return possible_actions[-1][0]

    def substract_predicates(self, origin, destination):
        """
        Subtracts 2 predicates, getting only the values that are different

        :param origin: Origin as Tuple[Predicate, Predicate, Predicate]
        :param destination: Destination as Tuple[Predicate, Predicate, Predicate]
        :return dict: Dict with the different values
        """
        #TODO: differentiate based on the discretizer complexity
        if type(origin) is str:
            origin = origin.split(',')
        if type(destination) is str:
            destination = destination.split(',')

        result = {}
        for value1, value2 in zip(origin, destination):
            if value1 != value2:
                result[value1.predicate] = (value1, value2)
        return result

    def nearby_predicates(self, state, greedy=False, verbose=False):
        """
        Gets nearby states from state

        :param verbose:
        :param greedy:
        :param state: State
        :return: List of [Action, destination_state, difference]
        """

        outs = self.out_edges(state, data=True)
        outs = [(u, d['action'], v, d['probability']) for u, v, d in outs]
        result = [(self.get_most_probable_option(v, greedy=greedy, verbose=verbose),
                   v,
                   self.substract_predicates(u, v)
                   ) for u, a, v, w in outs]
        
        result = sorted(result, key=lambda x: x[1])

        return result

    #modified
    def question3(self, predicate, action, greedy=False, verbose=False):
        #TODO: can be improved following the paper below (reward decomposition)
        #ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9659472
        """
        Answers the question: Why do you perform action X in state Y?
        """
        if verbose:
            print('***********************************************')
            print('* Why did not you perform X action in Y state?')
            print('***********************************************')

        if greedy:
            mode = PGBasedPolicyMode.GREEDY
        else:
            mode = PGBasedPolicyMode.STOCHASTIC
        pg_policy = PGBasedPolicy(self, mode)
        best_action = pg_policy.act_upon_discretized_state(predicate)
        result = self.nearby_predicates(predicate, greedy)
        explanations = []
        if verbose:
            print('I would have chosen:', best_action)
            print(f"I would have chosen {action} under the following conditions:")
        for a, v, diff in result:
            # Only if performs the input action
            #changed a == action in a.value == action
            if a.value == action:
                if verbose:
                    print(f"Hypothetical state: {v}")
                    for predicate_key,  predicate_value in diff.items():
                        print(f"   Actual: {predicate_key} = {predicate_value[0]} -> Counterfactual: {predicate_key} = {predicate_value[1]}")
                explanations.append(diff)
        if len(explanations) == 0 and verbose:
            print("\tI don't know where I would have ended up")
        return explanations


    def compute_reward_difference(self, state, action):
        """
        Computes the Reward Difference Explanation for selecting the given action.
        Args:
            state: discretized current velocity, position, and rotation
            #TODO: add objects
            action: the selected action for which to compute the explanation.

        Return:
            dict: a dictionary containing the difference in reward for the selected action
                compared to all other possible actions.
        
        Ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9659472
        """

        reward_difference = {}
        selected_reward = self.environment.compute_reward(state, action)

        for other_action in self.discretizer.all_actions():
            if other_action!= action:
                other_reward = self.environment.compute_reward(state, other_action)
                difference = tuple(selected - other for selected, other in zip(selected_reward, other_reward))
                reward_difference[other_action] = difference

        return reward_difference
    

    def question4(self, state, action, verbose=False):
        #NOTE: We use this metric to double check the reward function and see
        #whether it satisfies our desires. 
        #(otherwise, we could use it to answer the above question if we suppose drivers from the scenes are experts ?)

        if verbose:
            print('***********************************************')
            print('* why the vehicle in state s preferred to take the action a instead of all other possible actions?')
            print('***********************************************')

        reward_difference= self.compute_reward_difference(state, action)
        for other_action, difference in reward_difference.items():
            print(f"For {action} compared to {other_action}:")
            print(f"Speed Reward Difference: {difference[0]}")
            print(f"Safety Reward Difference: {difference[1]}")
            print(f"Smoothness Reward Difference: {difference[2]}")
            print(f"Progress Reward Difference: {difference[3]}")
            print("-------------------------------------------------------")


        


    ######################
    # SERIALIZATION
    ######################

    def _gram(self) -> str:
        graph_string = ''

        node_info = {
            node: {'id': i, 'value': self.discretizer.state_to_str(node),
                   'probability': self.nodes[node]['probability'],
                   'frequency': self.nodes[node]['frequency']}
            for i, node in enumerate(self.nodes)
        }
        # Get all unique actions in the PG
        action_info = {
            action: {'id': i, 'value': str(action)}
            for i, action in enumerate(set(action for _, _, action in self.edges))
        }

        for _, info in node_info.items():
            graph_string += f"\nCREATE (s{info['id']}:State " + "{" + f'\n  uid: "s{info["id"]}",\n  value: "{info["value"]}",\n  probability: {info["probability"]}, \n  frequency:{info["frequency"]}' + "\n});"
        for _, action in action_info.items():
            graph_string += f"\nCREATE (a{action['id']}:Action " + "{" + f'\n  uid: "a{action["id"]}",\n  value:{action["value"]}' + "\n});"

        for edge in self.edges:
            n_from, n_to, action = edge
            # TODO The identifier of an edge may need to be unique. Check and rework the action part of this if needed.
            graph_string += f"\nMATCH (s{node_info[n_from]['id']}:State) WHERE s{node_info[n_from]['id']}.uid = \"s{node_info[n_from]['id']}\" MATCH (s{node_info[n_to]['id']}:State) WHERE s{node_info[n_to]['id']}.uid = \"s{node_info[n_to]['id']}\" CREATE (s{node_info[n_from]['id']})-[:a{action_info[action]['id']} " + "{" + f"probability:{self[n_from][n_to][action]['probability']}, frequency:{self[n_from][n_to][action]['frequency']}" + "}" + f"]->(s{node_info[n_to]['id']});"

        return graph_string

    def _save_gram(self,
                   path: str
                   ):

        path_includes_gram = path[-5:] == '.gram'
        with open(f'{path}{"" if path_includes_gram else ".gram"}', 'w+') as f:
            f.write(self._gram())

    def _save_pickle(self,
                     path: str
                     ):

        path_includes_pickle = path[-7:] == '.pickle'
        with open(f'{path}{"" if path_includes_pickle else ".pickle"}', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_csv(self,
                  path_nodes: str,
                  path_edges: str,
                  path_trajectories: str
                  ):
        path_to_nodes_includes_csv = path_nodes[-4:] == '.csv'
        path_to_edges_includes_csv = path_edges[-4:] == '.csv'
        path_to_trajs_includes_csv = path_trajectories[-4:] == '.csv'

        node_ids = {}
        with open(f'{path_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['id', 'value', 'p(s)', 'frequency', 'is_destination'])
            for elem_position, node in enumerate(self.nodes):
                node_ids[node] = elem_position
                csv_w.writerow([elem_position, node,
                                self.nodes[node]['probability'], self.nodes[node]['frequency'], self.nodes[node]['is_destination']])

        with open(f'{path_edges}{"" if path_to_edges_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['from', 'to', 'action', 'p(s)', 'frequency'])
            for edge in self.edges:
                state_from, state_to, action = edge
                csv_w.writerow([node_ids[state_from], node_ids[state_to], action,
                                self[state_from][state_to][action]['probability'],
                                self[state_from][state_to][action]['frequency']])

        with open(f'{path_trajectories}{"" if path_to_trajs_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)

            for trajectory in self._trajectories_of_last_fit:

                csv_trajectory = []
                for elem_position, element in enumerate(trajectory):
                    # Process state
                    if elem_position % 2 == 0:
                        csv_trajectory.append(node_ids[element])
                    # Process action
                    else:
                        csv_trajectory.append(element)

                csv_w.writerow(csv_trajectory)

    # gram format doesn't save the trajectories
    def save(self,
             format: str,
             path: Union[str, List[str]]) \
            :
        if not self._is_fit:
            raise Exception('Policy Graph cannot be saved before fitting!')

        if format not in ['pickle', 'csv', 'gram']:
            raise NotImplementedError('format must be one of pickle, csv or gram')

        if format == 'csv':
            assert len(path) == 3, \
                "When saving in CSV format, path must be a list of 3 elements (nodes, edges, trajectories)!"
            self._save_csv(*path)
        elif format == 'gram':
            assert isinstance(path, str), "When saving in gram format, path must be a string!"
            self._save_gram(path)
        elif format == 'pickle':
            assert isinstance(path, str), "When saving in pickle format, path must be a string!"
            self._save_pickle(path)
        else:
            raise NotImplementedError



    ######################
    # POLICY ITERATION
    ######################

    def policy_iteration(self, scene_policy, gamma=0.99, theta=0.01, n_iter=30):
        '''
        Compute policy iteration algorithm on the policy graph build on a single (or subset) of scenes.

        Args:
            - scene_data: a policy graph based on a scene
            - gamma:
            - theta:

        Improves on the original total policy.
        '''
        is_policy_stable = False

        iteration_count = 0
        with tqdm.tqdm(total=None, desc='Policy Iteration Progress') as pbar:

            while not is_policy_stable:
                
                #policy evaluation
                V = PolicyGraph.policy_evaluation(scene_policy, gamma,theta)

                #policy improvement
                is_policy_stable, policy = PolicyGraph.policy_improvement(scene_policy, V, gamma)
                iteration_count += 1
                pbar.update(1)

                if iteration_count == n_iter:
                    is_policy_stable = True

        self._update_policy_graph(policy)
        self._is_fit = True
        print(f"Policy iteration completed in {iteration_count} iterations.")
    
    
    @staticmethod
    def policy_evaluation(scene_policy, gamma, theta):
        """
        Computes the value function for the current policy.
        """
        V = defaultdict(float)
        while True:
            delta = 0
            for state in scene_policy.nodes():#self.nodes():
                v = V[state]  
                V[state] = sum([scene_policy.get_edge_data(state, next_state, action)['probability']*
                                (scene_policy.get_edge_data(state, next_state, action)['reward'] + gamma * V[next_state])
                                for next_state in scene_policy[state] for action in scene_policy.get_edge_data(state, next_state)])
                delta = max(delta, abs(v-V[state]))
            if delta < theta:
                break
        return V
    
    @staticmethod
    def policy_improvement(scene_policy, V, gamma):
        """
        Updates the policy based on the value function.
        """
        policy_stable = True
        new_policy = defaultdict(lambda: defaultdict(float))
        for state in scene_policy.nodes():#self.nodes:
            
            current_best_action = None

            #compute action value for each action from this state
            action_values = {}
            for current_state, next_state, data in scene_policy.out_edges(state, data=True):              
                reward = data['reward']
                action = data['action']
                action_values[action] = reward + gamma*V[next_state]
            
            if not action_values:
                #print('No action starting from this state. Possibly the last state of the scene.')
                continue

            best_action = max(action_values, key=action_values.get)

            #update policy to choose the best action with probability 1
            #TODO: stochastic approach
            for action in action_values.keys():
                if action == best_action:
                    new_policy[state][action] = 1.0
                else:
                    new_policy[state][action] = 0.0
                
            if current_best_action != best_action:
                policy_stable = False
                      
        return policy_stable, new_policy
        
    def _update_policy_graph(self,policy):
        """
        Updates the policy graph based on the improved policy.
        """
        for state in policy:
            for action in policy[state]:
                #update edge probabilities based on the improved policy
                for next_state in self[state]:
                    if self.has_edge(state, next_state, key=action):
                        self[state][next_state][action]['probability']= policy[state][action]

    def _is_predicate_in_pg_and_usable(self, predicate) -> bool:
        return self.has_node(predicate) and len(self[predicate]) > 0






class PGBasedPolicyMode(Enum):
    GREEDY = auto()
    STOCHASTIC = auto()


class PGBasedPolicyNodeNotFoundMode(Enum):
    RANDOM_UNIFORM = auto()
    FIND_SIMILAR_NODES = auto()


class PGBasedPolicy(Agent):
    def __init__(self,
                 policy_graph: PolicyGraph,
                 mode: PGBasedPolicyMode,
                 node_not_found_mode: PGBasedPolicyNodeNotFoundMode = PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
                 ):
        
        
        self.pg = policy_graph
        self.dt=0.5
        self.wheel_base = 2.588 #in meters. Ref: https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        self.min_steer_angle = -7.7
        self.max_steer_angle = 6.3
        #self.visited_states = set()
        #self.newly_discovered_states = set()


        # Metrics of the PG Agent
        self.pg_metrics = {'AER': [], 'STD': [], 'ADE': [], 'FDE': []}
        
        assert mode in [PGBasedPolicyMode.GREEDY, PGBasedPolicyMode.STOCHASTIC], \
            'mode must be a member of the PGBasedPolicyMode enum!'
        self.mode = mode
        assert node_not_found_mode in [PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM,
                                       PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES], \
            'node_not_found_mode must be a member of the PGBasedPolicyNodeNotFoundMode enum!'
        self.node_not_found_mode = node_not_found_mode

        self.all_possible_actions = self._get_all_possible_actions()    
    
    
    def _get_all_possible_actions(self) -> Set[Any]:
        all_possible_actions = set()

        for node_from in self.pg:
            for node_to in self.pg[node_from]:
                for action in self.pg[node_from][node_to]:
                    all_possible_actions.add(action)

        return all_possible_actions

    def _get_action_probability_dist(self,
                                     predicate
                                     ) -> List[Tuple[int, float]]:
        # Precondition: self.pg.has_node(predicate) and len(self.pg[predicate]) > 0:
        
        # first be sure to apply to pg self.normalize()
        action_weights = defaultdict(lambda: 0)
        for dest_node in self.pg[predicate]:
            for action in self.pg[predicate][dest_node]:
                action_weights[action] += self.pg[predicate][dest_node][action]['probability']
        action_weights = [(a, action_weights[a]) for a in action_weights] #p(a, s)/p(s)
        return action_weights
    
    def _get_action(self,
                    action_weights: List[Tuple[int, float]]
                    ) -> int:
        if self.mode == PGBasedPolicyMode.GREEDY:
            sorted_probs: List[Tuple[int, float]] = sorted(action_weights, key=lambda x: x[1], reverse=True)
            return sorted_probs[0][0]
        elif self.mode == PGBasedPolicyMode.STOCHASTIC:
            p = [w for _, w in action_weights]
            return np.random.choice([a for a, _ in action_weights], p=[w for _, w in action_weights])
        else:
            raise NotImplementedError
    
    def act_upon_discretized_state(self, predicate):
        if self.pg.has_node(predicate) and len(self.pg[predicate]) > 0:
            action_prob_dist = self._get_action_probability_dist(predicate)
        else:
            if self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM:
                action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
            elif self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES:
                nearest_predicate = self.pg.get_nearest_predicate(predicate)
                if nearest_predicate is not None:  
                    action_prob_dist = self._get_action_probability_dist(nearest_predicate)
                    if action_prob_dist == []: #NOTE: we handle the case in which there is a nearest state, but this state has no 'next_state' (i.e. destination node of a scene)
                        action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
                else:
                    # Fallback if no nearest predicate is found
                    action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
            else:
                raise NotImplementedError
        return self._get_action(action_prob_dist) 

    def act(self,
            state
            ) -> Any:
        '''
        Args:
            Current continuous state.

        Output:
            Next action, given the input state.
            is_destination: 1 if input state is or is close to a (intermediate) destination state, 0 otherwise
        '''
        self.current_state = state
        
        x, y, speed, _, _, _, steering_angle = self.current_state
                    
        predicate = self.pg.discretizer.discretize( (x, y, speed,steering_angle) )
        return self.act_upon_discretized_state(predicate)


    def move(self, action: Action):
        """
        Updates the vehicle's state based on the current state and an action, using a simple physics model.
        The model assumes constant rates of change for speed and yaw.

        Args:
            self --> current_state (tuple): (x, y, speed, yaw, steering_angle) 
                - x: The x-coordinate of the vehicle's position.
                - y: The y-coordinate of the vehicle's position.
                - speed: The current speed of the vehicle.
                - yaw: The current yaw (orientation) of the vehicle in radians.
                - steering_angle: The current steering angle of the vehicle in radians.
            action: Action taken (e.g., "go straight", "turn", "gas", etc.). It influences acceleration and steering.

        Returns:
            tuple: Updated state (x', y', speed', yaw', steering_angle')

        ref: https://es.mathworks.com/help/mpc/ug/obstacle-avoidance-using-adaptive-model-predictive-control.html
        """

        #TODO: how will detected objects change their state?

        x, y, speed, yaw_rate, accel, yaw, steering_angle= self.current_state

        accel = 0
        steer = 0
        
        if action in (Action.TURN_LEFT, Action.GAS_TURN_LEFT, Action.BRAKE_TURN_LEFT):
            steer = 0.5
        elif action in (Action.TURN_RIGHT, Action.GAS_TURN_RIGHT, Action.BRAKE_TURN_RIGHT):
            steer = -0.5
        else: 
            steering_angle = 0
        
        if action in (Action.GAS, Action.GAS_TURN_LEFT, Action.GAS_TURN_RIGHT):
            accel += 0.001
        elif action in (Action.BRAKE, Action.BRAKE_TURN_LEFT, Action.BRAKE_TURN_RIGHT):
            accel -= 0.001
        
        if action is not Action.IDLE:
            speed_step = self.dt * accel
            yaw_step = self.dt * yaw_rate
            distance_step = self.dt * speed
        else:
            speed_step = 0
            yaw_step = 0
            distance_step = 0
            speed = 0

        # Update state
        x += distance_step * np.cos(yaw)
        y += distance_step * np.sin(yaw)
        speed = max(0, speed+speed_step) #ensure is non-negative

        yaw += yaw_step
        
        yaw_rate+=0# np.tan(steering_angle)*speed / self.wheel_base #TODO: fix
        
       
        steering_angle = max(self.min_steer_angle, min(self.max_steer_angle, steering_angle + steer*self.dt))

        yaw = max(-math.pi, yaw) if yaw<0 else min(math.pi,  yaw)
        
        # Integrate environment information
        #self.pg.environment.process_environment()
        
        return self.pg.environment.process_environment(x, y, speed, yaw_rate, accel, yaw, steering_angle)#x, y, speed, yaw_rate, accel, yaw, steering_angle


    

    
    '''
    def move_simple(self, action: Action):
        """
        Function that given current state and action returns the next continuous state.
        
        Args:
            self --> current_state (tuple): (x, y, v, steering_angle) where steering_angle is in radians
            action: Action taken (e.g., "go straight", "turn", "gas", etc.)
        
        Returns:
        tuple: Updated state (x', y', v', new_steering_angle)


        ref: https://es.mathworks.com/help/mpc/ug/obstacle-avoidance-using-adaptive-model-predictive-control.html

        """
        #TODO: how will detected objects change their state?
        x, y, velocity, _, _, _, theta = self.current_state

       
        if action in (Action.TURN_LEFT, Action.GAS_TURN_LEFT, Action.BRAKE_TURN_LEFT):
            steer_angle = +0.5
        if action in (Action.TURN_RIGHT, Action.GAS_TURN_RIGHT,  Action.BRAKE_TURN_RIGHT):
            steer_angle = -0.5
        else:
            steer_angle = 0
        
        #steer_angle_rad = np.deg2rad(steer_angle)

        #update velocity
        if action in (Action.GAS,  Action.GAS_TURN_LEFT,Action.GAS_TURN_RIGHT):
            velocity += 0.2 #self.discretizer.eps_vel ()
        elif action in (Action.BRAKE,  Action.BRAKE_TURN_LEFT, Action.BRAKE_TURN_RIGHT):
            velocity -= 0.2 #self.discretizer.eps_vel ()
        
        #update orientation (theta) based on steering angle and velocity
        #new_theta = theta - (velocity / self.wheel_base) * np.tan(steer_angle) * self.dt if action not in [Action.STRAIGHT, Action.GAS, Action.BRAKE] else theta
        new_theta = max(self.min_steer_angle, theta+steer_angle) if theta+steer_angle<0 else min(self.max_steer_angle,  theta + steer_angle)

        # Update position based on velocity and orientation
        delta_x = velocity * np.sin(new_theta) * self.dt
        delta_y = velocity * np.cos(new_theta) * self.dt

        # Calculate new position
        new_x = x + delta_x
        new_y = y + delta_y
    
        return (new_x, new_y, velocity,  _, _, _, new_theta)
        
    '''

    def step(self, action:Action) -> Tuple[Tuple[Predicate, Predicate, Predicate, Predicate], float, bool, str]:
        """
        Perform a step in the environment based on the computation of P(s'|s,a) as #(s,a,s')/(s,a).

        Args:
            action (Action): Action to be taken.
        Returns:
            next_state: tuple of predicates representing the next state after taking the action.
            reward: 

        """
        #update car state
        next_state = self.move(action)

        # compute reward
        
        #TODO review
        #predicate = self.pg.discretizer.discretize(self.current_state)
        x, y, speed, _, _, _, steering_angle = self.current_state   
        predicate = self.pg.discretizer.discretize( (x, y, speed,steering_angle) )
        reward = self.pg.environment.compute_reward(predicate, action)       
        # check if the episode is done
        done = False
        info = None

        return next_state, reward, done, info
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

    
    #################
    # TESTING
    #################

    def test_scene_trajectory(self, ground_truth_traj:list, max_steps=20, verbose = False, render = False):
        """
        Tests the the PGAgent in one scene.

        Args:
            ground_truth_traj: ground truth trajectory of the agent in a scene.
            data_file: csv file of states of the vehicle
            max_steps: number of steps of the episode
        """
        print('---------------------------------')
        print('* START TESTING')
        self.pg_metrics['AER'] = []
        self.pg_metrics['STD'] = []

        start_time = time.time()
        
        rewards = []

        pred_trajectory = []
            
        step_count = 0
        total_reward = 0
        state = ground_truth_traj.iloc[0].tolist()#self.pg.environment.reset(ground_truth_traj)
        
        #make sure ground truth and estimated trajectory are of same length
        ground_truth_traj = ground_truth_traj[:max_steps]

        while step_count < max_steps:
                    action_id = self.act(state)
                    action = self.pg.discretizer.get_action_from_id(action_id)
                    
                    next_state, reward, _, _ = self.step(action)
                    
                    total_reward +=sum(reward)
                    step_count +=1                    
                    
                    pred_trajectory.append([state[0], state[1]])

                    if verbose:
                        print('Actual state:', state)
                        print('Action:', action)
                    

                    state = next_state

        rewards.append(reward)
        
        # Compute the average reward and std
        #average_reward = np.sum(rewards) / len(scene_tokens)
        #std = np.std(rewards)

        #self.pg_metrics['AER'].append(average_reward)
        #self.pg_metrics['STD'].append(std)
        #self.epoch_mean_time = time.time() - start_time
        #print(f"Average Reward: {average_reward} and Standard Deviation: {std} --> Episode Mean Time: {self.epoch_mean_time}")

        ADE, FDE = PGBasedPolicy.distance_metrics(np.array(pred_trajectory),ground_truth_traj[['x','y']].to_numpy())
        self.pg_metrics['ADE'].append(ADE)
        self.pg_metrics['FDE'].append(FDE)
        
        self.epoch_mean_time = time.time() - start_time
        
        print(f"ADE: {ADE} and FDE: {FDE} --> Episode Mean Time: {self.epoch_mean_time}")


        print('* END TESTING')
        print('---------------------------------')
        print('* RESULTS')

        if render:
            #render PG agent path
            print('* PG agent path')
            self.pg.environment.render_egoposes_on_fancy_map(pred_trajectory)

            #render true path
            print('* True agent path')
            print(ground_truth_traj)
            self.pg.environment.render_egoposes_on_fancy_map(ground_truth_traj[['x','y']].values.tolist())






    def test(self, num_episodes, seed, data_file, max_steps=40, verbose = False, render = False):
        """
        Tests the the PGAgent in one scene.

        Args:
            num_episodes: n. of episodes to test
            seed: number to pseudo-randomly select a row in the data_file
            data_file: csv file of states of the vehicle
            max_steps: number of steps of the episode #TODO: should i provide the destination and make it stop when the destination arrives?
                    
        """
        print('---------------------------------')
        print('* START TESTING\n')
        self.pg_metrics['AER'] = []
        self.pg_metrics['STD'] = []

        start_time = time.time()
        
        #load file of states
        starting_points = pd.read_csv(data_file)
        rewards = []

        if render:
            trajectory = []
            
        for i in range(num_episodes):
            step_count = 0
            total_reward = 0
            state = self.pg.environment.reset(starting_points, seed)
            
            
            while step_count < max_steps:
                    action_id = self.act(state)
                    action = self.pg.discretizer.get_action_from_id(action_id)
                    
                    next_state, reward, _, _ = self.step(action)
                    
                    total_reward +=sum(reward)
                    step_count +=1
                    
                    

                    if verbose:
                        print('Actual state:', state)
                        print('Action:', action)
                    
                    if render:
                        trajectory.append([state[0], state[1]])

                    state = next_state

        rewards.append(reward)
        
        #self.env.close()

        # Compute the average reward and std
        average_reward = np.sum(rewards) / num_episodes
        std = np.std(rewards)

        self.pg_metrics['AER'].append(average_reward)
        self.pg_metrics['STD'].append(std)
        self.epoch_mean_time = time.time() - start_time
        print(f"Average Reward: {average_reward} and Standard Deviation: {std} --> Episode Mean Time: {self.epoch_mean_time}")


        print('* END TESTING')
        print('---------------------------------')
        print('* RESULTS')

        if render:
            self.pg.environment.render_egoposes_on_fancy_map(trajectory)



    #######################
    ### METRICS
    #######################
    @staticmethod
    def distance_metrics(estimated_trajectory, ground_truth):
        """
        Compute Average Displacement Error(ADE): average of pointwise L2 distances between the estimated trajectory and ground truth.
        Compute Final Displacement Error (FDE): the L2 distance between the final points of the estimation and the ground truth.

        estimated_trajectory: np.array of [x,y] values of the estimated trajectory
        ground_truth: np.array of [x,y] values of the true trajectory

        """
        ade =  np.mean(np.linalg.norm(estimated_trajectory - ground_truth, axis=1))

        final_gt = ground_truth[-1]
        final_pred = estimated_trajectory[-1]
        fde = np.linalg.norm(final_pred-final_gt)

        return ade, fde


    def compare(self):
        """
        Compares the metrics of the original RL agent vs the PG Agent.
        """
        try:
            agent_aer = sum(self.pg.agent_metrics['AER']) / len(self.pg.agent_metrics['AER'])
            agent_std = sum(self.pg.agent_metrics['STD']) / len(self.pg.agent_metrics['STD'])

            mdp_aer = sum(self.pg_metrics['AER']) / len(self.pg_metrics['AER'])
            mdp_std = sum(self.pg_metrics['STD']) / len(self.pg_metrics['STD'])

            transferred_learning = int((mdp_aer / agent_aer) * 100)
            diff_aer = -(agent_aer - mdp_aer)
            diff_std = -(agent_std - mdp_std)

            #percentage_new_states = (self.pg_metrics['new_state']/self.pg_metrics['Visited States'])
            #percentage_seen_states = (self.pg.number_of_nodes() / self.discretizer.get_num_possible_states())

            #percentage_std = (mdp_std /agent_std)

        except ZeroDivisionError:
            raise Exception('Agent Metrics are missing! Have you loaded the model?')

        print('---------------------------------')
        print('* COMPARATIVE')
        print('\t- Original RL Agent')
        print('\t\t+ Average Episode Reward:', agent_aer)
        print('\t\t+ Standard Deviation:', agent_std)
        print('\t- Policy Graph')
        print('\t\t+ Average Episode Reward:', mdp_aer)
        print('\t\t+ Standard Deviation:', mdp_std)
        #print('\t\t+ Num new states visited:', self.pg_metrics['new_state'])
        print('\t- Difference')
        print('\t\t+ Average Episode Reward Diff:', diff_aer)
        print('\t\t+ Standard Deviation Diff:', diff_std)
        print('\t- Transferred Learning:', transferred_learning, '%')
        #print('\t- Percentage New states:', percentage_new_states * 100, '%')
        #print('\t- Percentage Seen states:', percentage_seen_states * 100, '%')
        #print('\t- Percentage STD:', percentage_std * 100, '%')

        return (diff_aer,
                diff_std, transferred_learning)#, percentage_new_states, percentage_seen_states, percentage_std)

    
    """

    def get_state_transition(self, current_state, action):
        '''
        Function that computes next state based on compute max(P(s'|s,a))?
        '''
            
        frequency_s_a = sum(self.pg.get_edge_data(current_state, next_state, action)['frequency'] for next_state in self.pg[current_state])
        frequency_s_a_s = defaultdict(lambda: 0)

        for next_state in self.pg[current_state]:
            frequency_s_a_s[next_state] = self.pg[current_state][next_state][action]['frequency']
        
        next_state_distr = [(next_state, frequency_s_a_s[next_state]/frequency_s_a) for next_state in frequency_s_a_s]
        #TODO:return max value or sample it?
        return max(next_state_distr)
    """
