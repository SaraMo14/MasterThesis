import argparse
import numpy as np
import pgeon.policy_graph as PG
from pathlib import Path
from example.environment import SelfDrivingEnvironment
import pandas as pd
from example.transition import TransitionRecorded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help='The format of the Policy Graph to be loaded',
                        choices=['pickle', 'csv'])
    parser.add_argument('--policy-mode',
                        help='Whether to use the greedy or stochastic PG-based policy',
                        choices=['greedy', 'stochastic']) #TODO: original as well?
    parser.add_argument('--action-mode',
                        help='When visiting an unknown state, whether to choose action at random or based on what similar nodes chose.',
                        choices=['random', 'similar_nodes'])     
    parser.add_argument('--verbose', help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')

    args = parser.parse_args()
    input_format, policy_mode, node_not_found_mode, verbose = args.input, args.policy_mode, args.action_mode,args.verbose
    
    if args.policy_mode == 'greedy':
        mode = PG.PGBasedPolicyMode.GREEDY
    else:
        mode = PG.PGBasedPolicyMode.STOCHASTIC

    if args.action_mode == 'random':
        node_not_found_mode = PG.PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
    else:
        node_not_found_mode = PG.PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES
       
    
    if input_format == 'csv':
        nodes_path = f'example/dataset/data/policy_graphs/pg_nodes.csv'
        edges_path = f'example/dataset/data/policy_graphs/pg_edges.csv'
        env = SelfDrivingEnvironment()
        pg = PG.PolicyGraph.from_nodes_and_edges(nodes_path, edges_path, env, env.discretizer )
    else:
        pg = PG.PolicyGraph.from_pickle(f'example/dataset/data/policy_graphs/pg.pickle')

    agent = PG.PGBasedPolicy(pg, mode, node_not_found_mode)

    if verbose:
        print(f'Successfully loaded Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')
        print(f'Policy mode: {policy_mode}')
        print(f'Node not found mode: {node_not_found_mode}')
        print()

    initial_state = np.array([329.6474941596216, 660.1966888688361, 5.108549775006556, 2.8287073423413176,])
    final_state = np.array([309.1878607795861, 668.6677254778403,  7.946119310024729e-05, 2.678093668689712,])
    tot_reward, reached_destination = env.compute_total_reward(agent, initial_state, final_state, max_steps=100)
    
    print(f'Final scene reward: {tot_reward}')
    print(f'Destination reached: {reached_destination}')
    #python3 test_pg.py --input csv --policy-mode greedy --action-mode random --verbose 