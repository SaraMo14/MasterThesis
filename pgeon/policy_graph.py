from collections import defaultdict
from enum import Enum, auto
from typing import Dict, Tuple, Any, List, Union, Set
import csv
import pickle
import gymnasium as gym
import networkx as nx
import numpy as np
import tqdm
from pgeon.agent import Agent
from pgeon.discretizer import Discretizer


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

        self.unique_states: Dict[str, int] = {}
        self.state_to_be_discretized = ['x', 'y', 'velocity', 'steering_angle'] #yaw not needed 
        self.state_columns_for_action = ['delta_local_x', 'delta_local_y', 'velocity', 'acceleration', 'steering_angle'] #heading_change_rate not needed
     
        # Metrics of the original Agent
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
                     agent: Agent,
                     max_steps: int = None,
                     seed: int = None
                     ) -> List[Any]:

        observation, _ = self.environment.reset(seed=seed)
        done = False
        trajectory = [self.discretizer.discretize(observation)]

        step_counter = 0
        while not done:
            if max_steps is not None and step_counter >= max_steps:
                break

            action = agent.act(observation)
            observation, _, done, done2, _ = self.environment.step(action)
            done = done or done2

            trajectory.extend([action, self.discretizer.discretize(observation)])

            step_counter += 1

        return trajectory
    
    
    

    def _update_with_trajectory(self,
                                trajectory: List[Any]
                                ):

        # Only even numbers are states
        states_in_trajectory = [trajectory[i] for i in range(len(trajectory)) if i % 2 == 0]
        all_new_states_in_trajectory = {state for state in set(states_in_trajectory) if not self.has_node(state)}
        self.add_nodes_from(all_new_states_in_trajectory, frequency=0)

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



    def compute_trajectory(self, states, verbose = False):
        """
            Discretizes a trajectory (list of states) and stores unique states and actions.

            Args:
                states: DataFrame containing state information for each time step.

            Returns:
                List containing tuples of (current state ID, action ID, next state ID).
        """
        trajectory = []

        self.detection_cameras = [col for col in states.columns if 'CAM' in col] #NOTE: this should be assigned even when testing the PG (not only when computing the trajectory)
        
        n_states = len(states)

        rewards = []
        for i in range(n_states-1):
            
            # discretize current state
            current_state_to_discretize = states.iloc[i][self.state_to_be_discretized].tolist()
            current_detection_info = states.iloc[i][self.detection_cameras] if len(self.detection_cameras)>0 else None
            discretized_current_state = self.discretizer.discretize(current_state_to_discretize, current_detection_info)
            current_state_str = self.discretizer.state_to_str(discretized_current_state)
            current_state_id = self.add_unique_state(current_state_str)
            
            #check if is scene destination state
            current_scene = states.iloc[i]['scene_token']
            next_scene = states.iloc[i+1]['scene_token']
            if current_scene != next_scene:
                action_id = None
            else:
                # Determine action based on the full state information
                current_state_for_action = states.iloc[i][self.state_columns_for_action].tolist()
                next_state_for_action = states.iloc[i+1][self.state_columns_for_action].tolist()
                action = self.discretizer.determine_action(current_state_for_action, next_state_for_action)
                action_id = self.discretizer.get_action_id(action)
    
            if verbose:
                # From 'predicate', choosing 'act' we achieved state 'predicate_next'
                print('From', discretized_current_state, ' -> ',action, ' -> ', next_state_for_action)


            trajectory.extend([current_state_id, action_id])        
        
        #add last state
        last_state_to_discretize = states.iloc[n_states-1][self.state_to_be_discretized].tolist()
        last_state_detections = states.iloc[n_states-1][self.detection_cameras] if len(self.detection_cameras)>0 else None
        discretized_last_state = self.discretizer.discretize(last_state_to_discretize, last_state_detections)
        last_state_str = self.discretizer.state_to_str(discretized_last_state)
        last_state_id = self.add_unique_state(last_state_str)
        trajectory.extend([last_state_id, None, None])        
        
        return trajectory

    def add_unique_state(self, state_str: str) -> int:
        if state_str not in self.unique_states:
            self.unique_states[state_str] = len(self.unique_states)
        return self.unique_states[state_str]


    def test(self, scenes, agent, verbose = False, max_steps = 100):
        """
        Tests the PGAgent in the given scenes (episodes).

        Args:
            scenes: list of (scene_id, scene_start_state, scene_end_state)
            agent: PolicyBasedAgent
            verbose:
                    
        """
        print('---------------------------------')
        print('* START TESTING\n')

        self.pg_metrics['AER'] = []
        self.pg_metrics['STD'] = []

        #start_time = time.time()

        #rewards = []

        for scene_id, initial_state, final_state in scenes:
            #self.env.reset()
            reached_final = False
            step_count = 0
            total_reward = 0

            self.current_state = initial_state
        
            while step_count < max_steps:
                action_id, is_destination = agent.act(self.current_state)
                action = self.discretizer.get_action_from_id(action_id)
                next_state, reward, _, _ = self.env.step(action, is_destination)
                total_reward +=reward
                
                self.current_state = next_state
                
                print(f'step count: {step_count}')
                step_count +=1

                if next_state == final_state: #TODO: fix
                    reached_final = True
                    break
        
        # Once we finished the feeding, then we build the Graph
        print('* END TESTING')
        print('---------------------------------')
        print('* RESULTS')
        #print('\t- Average Reward:', sum(self.pg_metrics['AER']) / len(self.pg_metrics['AER']))
        #print('\t- Standard Deviation:', sum(self.pg_metrics['STD']) / len(self.pg_metrics['STD']), '\n')


    
    #TODO:  update
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
        best_action = pg_policy.act_upon_discretized_state(predicate)[0]
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
                csv_w.writerow([elem_position, self.discretizer.state_to_str(node),
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
            - thetha:

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
        #self.visited_states = set()
        #self.newly_discovered_states = set()



        
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

    '''
    def _is_predicate_in_pg_and_usable(self, predicate) -> bool:
        return self.pg.has_node(predicate) and len(self.pg[predicate]) > 0

    def _get_nearest_predicate(self,
                               predicate, verbose=False
                               ):
        nearest_state_generator = self.pg.discretizer.nearest_state(predicate)
        new_predicate = predicate
        try:
            while not self.pg._is_predicate_in_pg_and_usable(new_predicate):
                new_predicate = next(nearest_state_generator)
        except StopIteration:
            print("No nearest states available.")
            new_predicate = None
        return new_predicate
    '''

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
    
    #TODO: fix how we handle is_destination
    def act_upon_discretized_state(self, predicate):
        is_destination = False
        if self.pg.has_node(predicate) and len(self.pg[predicate]) > 0:
            action_prob_dist = self._get_action_probability_dist(predicate)
            is_destination = self.pg.nodes[predicate]['is_destination']
        else:
            if self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM:
                action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
            elif self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES:
                nearest_predicate = self.pg.get_nearest_predicate(predicate)
                if nearest_predicate is not None:  
                    action_prob_dist = self._get_action_probability_dist(nearest_predicate)
                    if action_prob_dist == []: #NOTE: we handle the case in which there is a nearest state, but this state has no 'next_state' (i.e. destination node of a scene)
                        action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
                    
                    is_destination = self.pg.nodes[nearest_predicate]['is_destination']
                else:
                    # Fallback if no nearest predicate is found
                    action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
            else:
                raise NotImplementedError
        return self._get_action(action_prob_dist), is_destination 

    def act(self,
            state
            ) -> Any:
        '''
        Args:
            Current continuous state.

        Output:
            Next action, given the input state.
            is_destination: 1 if input state is a (intermediate) destination state, 0 otherwise
        '''
        predicate = self.pg.discretizer.discretize(state)
        return self.act_upon_discretized_state(predicate)


   
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