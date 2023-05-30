import os
import pickle
import numpy as np
from ..utils.transforms import apply_transform


class Loader:
    def __init__(self,
                 data_root,
                 metadata,
                 map_frame_id=True,
                 load_scene=True,
                 load_objects=True,
                 load_feats=['xyz', 'feats'],
                 load_dims=[3, 2],
                 load_fraction=1.0,
                 use_dims=None,
                 obj_min_pts=None,
                 max_pts=None,
                 to_ego_frame=True,
                 relative_timestamp=True,
                 timestamp_dim=-1):

        if isinstance(metadata, str):
            with open(metadata, 'rb') as f:
                metadata = pickle.load(f)

        if not isinstance(metadata, dict):
            raise TypeError(f'metadata must be a dict, got {type(metadata)}')

        self.data_root = data_root
        self.map_frame_id = map_frame_id
        self.scene_infos = metadata.get('scene_infos', dict())
        self.obj_infos = metadata.get('obj_infos', dict())
        self.frame_infos = metadata.get('frame_infos', dict())
        self.frame_id_map = metadata.get('frame_id_map', dict())
        self.load_scene = load_scene
        self.load_objects = load_objects
        self.load_feats = load_feats
        self.load_dims = load_dims
        self.load_fraction = load_fraction
        self.use_dims = use_dims
        self.obj_min_pts = obj_min_pts
        self.max_pts = max_pts
        self.to_ego_frame = to_ego_frame
        self.relative_timestamp = relative_timestamp
        self.timestamp_dim = timestamp_dim

    def get_frame_info(self, frame_id):
        if self.map_frame_id and frame_id in self.frame_id_map:
            frame_id = self.frame_id_map[frame_id]

        if frame_id not in self.frame_infos:
            raise ValueError(f'frame {frame_id} does not exist in metadata')

        return self.frame_infos[frame_id]

    def load_points(self, info, min_pts=None):
        points = []
        if 'pts_data' in info:
            for name in self.load_feats:
                points.append(info['pts_data'][f'pts_{name}'])
        elif 'path' in info:
            path = info['path']
            for name, dim in zip(self.load_feats, self.load_dims):
                feats_file = f'{self.data_root}/{path}/pts_{name}.bin'
                num_pts = int(os.stat(feats_file).st_size // (4 * dim))
                if min_pts is not None and num_pts < min_pts:
                    break
                num_pts -= int(num_pts * self.load_fraction)
                points.append(np.fromfile(feats_file,
                                          offset=4 * dim * num_pts,
                                          dtype=np.float32).reshape(-1, dim))
        else:
            raise ValueError(f'info must have either path or pts_data')
        if len(points) == 0:
            return np.zeros((0, np.sum(self.load_dims)), dtype=np.float32)
        else:
            return np.concatenate(points, axis=-1)

    def load(self, frame_id):
        if isinstance(frame_id, dict):
            frame_info = frame_id
        else:
            frame_info = self.get_frame_info(frame_id)

        points = []

        if self.load_scene:
            scene_id = frame_info.get('scene_id', None)
            if scene_id not in self.scene_infos:
                raise ValueError(
                    f'scene_id {scene_id} does not exist in metadata')
            points.append(self.load_points(self.scene_infos[scene_id]))

        if self.load_objects:
            obj_poses = frame_info.get('obj_poses', dict())
            for obj_id, obj_pose in obj_poses.items():
                if obj_id not in self.obj_infos:
                    raise ValueError(
                        f'obj_id {obj_id} does not exist in metadata')
                min_pts = None
                if self.obj_min_pts is not None:
                    if isinstance(self.obj_min_pts, dict) and \
                            self.obj_infos[obj_id]['type'] in self.obj_min_pts:
                        min_pts = self.obj_min_pts[self.obj_infos[obj_id]['type']]
                    else:
                        min_pts = self.obj_min_pts
                obj_pts = self.load_points(self.obj_infos[obj_id], min_pts)
                obj_pts[:,:3] = apply_transform(obj_pose, obj_pts[:,:3])
                points.append(obj_pts)

        if len(points) > 0:
            points = np.concatenate(points)
        else:
            points = np.zeros((0, sum(self.load_dims)), dtype=np.float32)

        if self.max_pts and points.shape[0] > self.max_pts:
            indices = np.random.choice(points.shape[0], self.max_pts, replace=False)
            points = points[indices]

        if isinstance(self.use_dims, int):
            points = points[:,:self.use_dims]
        elif isinstance(self.use_dims, (list, tuple)):
            points = points[:,self.use_dims]

        if self.to_ego_frame:
            ego_pose = frame_info.get('ego_pose', np.eye(4))
            points[:,:3] = apply_transform(np.linalg.inv(ego_pose), points[:,:3])

        if self.relative_timestamp and self.timestamp_dim < points.shape[1]:
            points[:,self.timestamp_dim] -= frame_info.get('timestamp', 0)

        return points

    def __getitem__(self, frame_id):
        return self.load(frame_id)

    def __call__(self, frame_id):
        return self.load(frame_id)
