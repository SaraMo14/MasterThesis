import argparse
import numpy as np
import pgeon.policy_graph as PG
from pathlib import Path
from example.environment import SelfDrivingEnvironment
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help='The format of the Policy Graph to be loaded',
                        choices=['pickle', 'csv'])
    parser.add_argument('--policy-mode',
                        help='Whether to use the greedy or stochastic PG-based policy',
                        choices=['greedy', 'stochastic'])
    parser.add_argument('--action-mode',
                        help='When visiting an unknown state, whether to choose action at random or based on what similar nodes chose.',
                        choices=['random', 'similar_nodes'])     
    parser.add_argument('--verbose', help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')

    #TODO: give in input the discretizer or the complexity
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
       
    
    #source_folder = ''

    if input_format == 'csv':
        nodes_path = f'example/dataset/data/policy_graphs/pg_nodes.csv'
        edges_path = f'example/dataset/data/policy_graphs/pg_edges.csv'
        env = SelfDrivingEnvironment() #TODO: give in input the discretizer

        pg = PG.PolicyGraph.from_nodes_and_edges(nodes_path, edges_path, env, env.discretizer )
    else:
        pg = PG.PolicyGraph.from_pickle(f'example/dataset/data/policy_graphs/pg.pickle')

    agent = PG.PGBasedPolicy(pg, mode, node_not_found_mode)

    if verbose:
        print(f'Successfully loaded Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')
        print(f'Policy mode: {policy_mode}')
        print(f'Node not found mode: {node_not_found_mode}')
        print()


    #static metrics
    #evaluator = PolicyGraphEvaluator(pg)
    #entropy_metrics_graph = evaluator.compute_entropy_metrics()
    #for state, metrics in entropy_metrics_graph.items():
    #    print(f"State: {state}, H(s): {metrics['Hs']:.2f}, Ha(s): {metrics['Ha']:.2f}, Hw(s): {metrics['Hw']:.2f}")


    initial_state = np.array([329.6474941596216, 660.1966888688361, 5.108549775006556, -0.2023862177193762])
    data = {
    'detect_CAM_BACK': {('vehicle.truck', 'vehicle.moving'): 1, ('vehicle.car', 'vehicle.moving'): 1, ('vehicle.car', 'vehicle.parked'): 2},
    'detect_CAM_FRONT': {('vehicle.bus.rigid', 'vehicle.moving'): 1, ('vehicle.construction', 'vehicle.parked'): 1}
        }
    objects = pd.Series(data)
    
    final_state = np.array([309.1878607795861, 668.6677254778403,  7.946119310024729e-05, -1.2007145498601297])

    disc_initial_state = pg.discretizer.discretize(initial_state, objects)
    disc_final_state = pg.discretizer.discretize(final_state)

    print(f'Initial discretized state: {disc_initial_state}')
    print(f'Final discretized state: {disc_final_state}')
    print('-----------------------------------------------------------')
    
    tot_reward, reached_destination = env.compute_total_reward(agent, disc_initial_state, disc_final_state, max_steps=600)
    
    print(f'Final scene reward: {tot_reward}')
    print(f'Destination reached: {reached_destination}')

    #with open(f'example/dataset/data/rewards/rewards_{args.policy_mode}.csv', 'w+') as f:
    #    csv_w = csv.writer(f)
    #    csv_w.writerow(tot_reward)
    
    
    #python3 test_pg.py --input csv --policy-mode greedy --action-mode random --verbose 
