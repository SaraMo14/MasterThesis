import argparse
import pgeon.policy_graph as PG
from pathlib import Path
from example.environment import SelfDrivingEnvironment
import pandas as pd
from example.transition import TransitionRecorded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input data folder of states and actions.', default=".")
    parser.add_argument('--normalize', help='Whether the probabilities are stored normalized or not',
                        action='store_true')
    parser.add_argument('--output', help='Which format to output the Policy Graph',
                        default='csv', choices=['pickle', 'csv', 'gram'])
    parser.add_argument('--verbose', help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')
    

    args = parser.parse_args()
    data_folder, verbose, normalize, output_format = args.input, args.verbose, args.normalize, args.output

    # Generate Policy Graph
    env = SelfDrivingEnvironment()

    #from existing csv file
    #pg = PG.PolicyGraph.from_nodes_and_edges(str(Path(data_folder) / 'nuscenes_nodes.csv'), str(Path(data_folder) / 'nuscenes_edges.csv'), env, env.discretizer  )

    #from raw data
    dtype_dict = {
        'modality': 'category',  # for limited set of modalities, 'category' is efficient
        'scene_token': 'str',  
        'steering_angle': 'float64',  
        'timestamp': 'str',  # To enable datetime operations
        'rotation': 'object',  # Quaternion (lists)
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
        'yaw': 'float64',  
        'velocity': 'float64',
        'acceleration': 'float64',
        'heading_change_rate': 'float64',
        'delta_local_x': 'float64',
        'delta_local_y': 'float64'
    }
    df = pd.read_csv(Path(data_folder) / 'dataset_v1.0-trainval_lidar_0.csv', dtype=dtype_dict, parse_dates=['timestamp'])
    trajectory = env.discretizer.compute_trajectory(df)
    recorder = TransitionRecorded()
    recorder.process_transitions(trajectory)
    states_df, actions_df  = recorder.create_dataframes(env.discretizer.unique_states)
    pg = PG.PolicyGraph.from_data_frame(states_df, actions_df, env, env.discretizer)

    
    if normalize:
        pg._normalize()
        
    #normalize = "norm" if normalize else ""
    
    if output_format == 'csv':
        nodes_path = f'example/dataset/data/policy_graphs/pg_nodes.{output_format}'
        edges_path = f'example/dataset/data/policy_graphs/pg_edges.{output_format}'
        traj_path = f'example/dataset/data/policy_graphs/pg_traj.{output_format}'
        pg.save(output_format, [nodes_path, edges_path, traj_path])
    else:
        pg.save(output_format, f'example/dataset/data/policy_graphs/pg.{output_format}')


    if verbose:
        print(f'Successfully generated Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')

    #python3 generate_pg.py --input /home/saramontese/Desktop/MasterThesis/example/dataset/data/sets/nuscenes --normalize --verbose
