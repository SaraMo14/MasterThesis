import argparse
import numpy as np
import pgeon.policy_graph as PG
from pathlib import Path
from example.environment import SelfDrivingEnvironment
from example.discretizer.discretizer import AVDiscretizer
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_id',
                        help='The id of the Policy Graph to be loaded')
    parser.add_argument('--test_set',
                        help="csv file containg the test set of the preprocessed nuScenes dataset.")
    parser.add_argument('--policy-mode',
                        help='Whether to use the original agent, or a greedy or stochastic PG-based policy',
                        choices=['original','greedy', 'stochastic'])
    parser.add_argument('--action-mode',
                        help='When visiting an unknown state, whether to choose action at random or based on what similar nodes chose.',
                        choices=['random', 'similar_nodes'])     
    parser.add_argument('--discretizer', type=int, 
                        help='Specify the discretizer of the input data.', 
                        choices=[0, 1, 2], default=0)
    parser.add_argument('--city', 
                        help='Specify city to consider when testing the PG.', 
                        choices=['b','s1','s2', 's3'], 
                        default="b")
    #NOTE: currently only one city at the time can be tested
    parser.add_argument('--verbose', 
                        help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')

    
    args = parser.parse_args()
    training_id, city_id, discretizer_id, test_set, verbose = args.training_id, args.city, args.discretizer, args.test_set, args.verbose
    

    if city_id == 'b': 
        city = 'boston-seaport'
    elif city_id == 's1':
        city = 'singapore-hollandvillage'
    elif city_id == 's2':
        city = 'singapore-onenorth'
    elif city_id == 's3':
        city = 'singapore-queenstown'
  

    #load test set
    dtype_dict = {
        'modality': 'category',  # for limited set of modalities, 'category' is efficient
        'scene_token': 'str',  
        'steering_angle': 'float64', 
        'location': 'str',
        'timestamp': 'str',  # To enable datetime operations
        'rotation': 'object',  # Quaternion (lists)
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
        'yaw': 'float64',  
        'velocity': 'float64',
        'acceleration': 'float64',
        'yaw_rate': 'float64'
    }
    test_df = pd.read_csv(Path('example/dataset/data/sets/nuscenes') / test_set, dtype=dtype_dict, parse_dates=['timestamp'])
    test_df = test_df[test_df['location'] == city]
    



    if args.policy_mode == 'greedy':
        mode = PG.PGBasedPolicyMode.GREEDY
    else:
        mode = PG.PGBasedPolicyMode.STOCHASTIC

    if args.action_mode == 'random':
        node_not_found_mode = PG.PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
    else:
        node_not_found_mode = PG.PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES
        
        
    #load PG-based agent
    nodes_path = f'example/dataset/data/policy_graphs/{training_id}_nodes.csv'
    edges_path = f'example/dataset/data/policy_graphs/{training_id}_edges.csv'
        
    environment = SelfDrivingEnvironment(city)
    discretizer = {0: AVDiscretizer, 1: None, 2: None}[discretizer_id](environment)

    pg = PG.PolicyGraph.from_nodes_and_edges(nodes_path, edges_path, environment, discretizer)
    agent = PG.PGBasedPolicy(pg, mode, node_not_found_mode)
    
    if verbose:
        print(f'Successfully loaded Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')
        print(f'Policy mode: {args.policy_mode}')
        print(f'Node not found mode: {node_not_found_mode}')
        print()

    original_agent = True if args.policy_mode == 'original' else False
    avg_nll_action, std_nll_action, avg_nll_world, std_nll_world = agent.compute_static_metrics(test_set=test_df, verbose = verbose, original_agent=original_agent)
    #with open(f'example/dataset/data/rewards/rewards_{args.policy_mode}.csv', 'w+') as f:
    #    csv_w = csv.writer(f)
    #    csv_w.writerow(tot_reward)

 


    #static metrics
    #evaluator = PolicyGraphEvaluator(pg)
    #entropy_metrics_graph = evaluator.compute_entropy_metrics()
    #for state, metrics in entropy_metrics_graph.items():
    #    print(f"State: {state}, H(s): {metrics['Hs']:.2f}, Ha(s): {metrics['Ha']:.2f}, Hw(s): {metrics['Hw']:.2f}")

    
    #TODO specify test_city
  
    
    #python3 test_pg.py --training_id pg_Call_D0 --test_set 'test_v1.0-mini_lidar_0.csv' --policy-mode greedy --action-mode random --discretizer 0 --city 'b' --verbose