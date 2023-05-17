import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm.autonotebook import tqdm
from lamtk.aggregation import NuScenesAggregator, ObjectAggregator, SceneAggregator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create complete nuScenes dataset')
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
        default='./data/lamtk/complete/nuscenes',
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
        data_pipeline = [
            'load_frame_info',
            'load_points',
            'load_pts_labels',
            'load_obj_boxes3d',
            'load_obj_labels',
            'extract_obj_points'
        ],
        data_fields = [
            'frame_id', 'timestamp', 'ego_pose',
            'obj_pts_xyz', 'obj_pts_feats', 'obj_pts_dir',
            'obj_pts_range', 'obj_pts_labels',
            'bg_pts_xyz', 'bg_pts_feats', 'bg_pts_dir',
            'bg_pts_range', 'bg_pts_labels',
            'obj_labels', 'obj_boxes3d',
        ],
        obj_aggregator = ObjectAggregator(
            keep_labels=set(np.arange(2,24).tolist())),
        scene_aggregator = SceneAggregator(
            keep_labels=set(np.arange(24,31).tolist())),
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
        obj_infos = results_i['obj_infos']
        frame_infos = results_i['frame_infos']

        scene_dir = out_dir / 'scenes' / scene_id
        os.makedirs(scene_dir, exist_ok=True)
        scene_info['path'] = Path('scenes') / scene_id
        scene_pts_data = scene_info.pop('pts_data')
        for name, data in scene_pts_data.items():
            data.astype(np.float32).tofile(scene_dir / f'{name}.bin')
        results['scene_infos'][scene_id] = scene_info

        for obj_id, obj_info in obj_infos.items():
            obj_dir = out_dir / 'objects' / obj_id
            os.makedirs(obj_dir, exist_ok=True)
            obj_info['path'] = Path('objects') / obj_id
            obj_pts_data = obj_info.pop('pts_data')
            for name, data in obj_pts_data.items():
                data.astype(np.float32).tofile(obj_dir / f'{name}.bin')
            results['obj_infos'][obj_id] = obj_info

        results['frame_infos'].update(frame_infos)

    metadata_dir = out_dir / 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_suffix = f'trainval_{args.begin:0>3}-{args.end:0>3}'
    metadata_path = metadata_dir / f'metadata_{metadata_suffix}.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(results, f)
