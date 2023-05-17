import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm.autonotebook import tqdm
from lamtk.aggregation import NuScenesAggregator, SceneAggregator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create aggregated nuScenes dataset')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuscenes',
        required=False,
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        required=False,
        help='specify the split of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/lamtk/agg/nuscenes',
        required=False,
        help='the output directory')
    parser.add_argument(
        '--begin',
        type=int,
        default=0,
        required=False,
        help='')
    parser.add_argument(
        '--end',
        type=int,
        default=850,
        required=False,
        help='')
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        required=False,
        help='the rank of the process')
    args = parser.parse_args()

    agg = NuScenesAggregator(
        version=args.version,
        dataroot=args.dataroot,
        data_pipeline = ['load_frame_info', 'load_points',
                         'extract_obj_points'],
        data_fields = ['frame_id', 'ego_pose', 'timestamp',
                       'bg_pts_xyz', 'bg_pts_feats'],
        obj_aggregator = None,
        scene_aggregator = SceneAggregator(ds_voxel_size=0.05,
                                           ds_average=True,
                                           remove_outliers=False,
                                           estimate_normals=False),
        split_scene=False,
        rank=args.rank
    )

    results = dict(scene_infos=dict(), obj_infos=dict(), frame_infos=dict())
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for idx in tqdm(range(args.begin, args.end), desc=f'[{args.rank}] Aggregating nuScenes dataset', position=2*args.rank):
        if idx >= len(agg.scene_ids):
            break

        scene_id = agg.scene_ids[idx]
        results_i = agg[scene_id]
        scene_info = results_i['scene_info']
        frame_infos = results_i['frame_infos']

        scene_dir = out_dir / 'scenes' / scene_id
        os.makedirs(scene_dir, exist_ok=True)
        scene_info['path'] = Path('scenes') / scene_id
        scene_pts_data = scene_info.pop('pts_data')
        for name, data in scene_pts_data.items():
            data.astype(np.float32).tofile(scene_dir / f'{name}.bin')
        results['scene_infos'][scene_id] = scene_info
        results['frame_infos'].update(frame_infos)

    metadata_dir = out_dir / 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_suffix = f'trainval_{args.begin:0>3}-{args.end:0>3}'
    metadata_path = metadata_dir / f'metadata_{metadata_suffix}.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(results, f)
