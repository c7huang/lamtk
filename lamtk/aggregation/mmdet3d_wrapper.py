import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core import LiDARPoints, LiDARInstance3DBoxes
from .loader import Loader
from ..utils.transforms import affine_transform


@PIPELINES.register_module()
class LoadAggregatedPoints:
    def __init__(self,
                 data_root,
                 metadata_path,
                 dbinfos_path=None,
                 load_scene=True,
                 load_objects=True,
                 load_feats=['xyz', 'feats'],
                 load_dims=[3, 2],
                 load_fraction=1.0,
                 box_origin=[0.5, 0.5, 0.5],
                 use_dims=None,
                 load_as='points',
                 load_format='mmdet3d',
                 replace_existing=False):
        self.loader = Loader(
            data_root,
            metadata_path,
            load_scene=load_scene,
            load_objects=load_objects,
            load_feats=load_feats,
            load_dims=load_dims,
            load_fraction=load_fraction,
            to_ego_frame=True,
        )
        self.box_origin_shift = torch.as_tensor(box_origin, dtype=torch.float32) - \
            torch.as_tensor([0.5, 0.5, 0.0], dtype=torch.float32)
        self.use_dims = use_dims
        if not isinstance(load_as, (tuple, list)):
            load_as = [load_as]
        self.load_as = load_as
        self.load_format = load_format
        self.replace_existing = replace_existing

        # group id (unique in dbinfos) -> object id mapping
        self.gid2oid = dict()
        if dbinfos_path is not None:
            with open(dbinfos_path, 'rb') as f:
                dbinfos = pickle.load(f)
            for cls_infos in dbinfos.values():
                for obj_info in cls_infos:
                    if obj_info['group_id'] in self.gid2oid:
                        raise ValueError('invalid dbinfos: group_id is not unique')
                    frame_info = self.loader.get_frame_info(obj_info['image_idx'])
                    self.gid2oid[obj_info['group_id']] = \
                        frame_info['obj_ids'][obj_info['gt_idx']]

    def get_frame_info(self, input_dict):
        frame_info = self.loader.get_frame_info(input_dict['sample_idx'])
        aug_transform = np.identity(4)

        if 'gt_bboxes_3d' in input_dict:
            gt_boxes = input_dict['gt_bboxes_3d']
            gt_boxes = LiDARInstance3DBoxes(
                tensor = gt_boxes.tensor.clone(), 
                box_dim = gt_boxes.box_dim, 
                with_yaw = gt_boxes.with_yaw)

            # Shift box origin based on the aggregated point cloud format
            gt_boxes.tensor[:,:3] += self.box_origin_shift * gt_boxes.tensor[:,3:6]

            # Handle transformation augmentations
            # 1. undo all the transformations to the bounding boxes
            # 2. aggregate the final transformation matrix `aug_transformation`
            if 'transformation_3d_flow' in input_dict:
                for t in reversed(input_dict['transformation_3d_flow']):
                    if t == 'R':
                        pcd_rotation = input_dict['pcd_rotation'].T
                        pcd_rotation_angle = np.arctan2(pcd_rotation[0,1], pcd_rotation[0,0])
                        # Undo augmentation
                        gt_boxes.rotate(-pcd_rotation_angle)
                        # Record augmentation
                        aug_transform = aug_transform @ affine_transform(
                            matrix=pcd_rotation)
                    elif t == 'S':
                        pcd_scale_factor = input_dict['pcd_scale_factor']
                        # Undo augmentation
                        gt_boxes.scale(1/pcd_scale_factor)
                        # Record augmentation
                        aug_transform = aug_transform @ affine_transform(
                            matrix=np.identity(3)*pcd_scale_factor)
                    elif t == 'T':
                        pcd_trans = input_dict['pcd_trans']
                        # Undo augmentation
                        gt_boxes.translate(-pcd_trans)
                        # Record augmentation
                        aug_transform = aug_transform @ affine_transform(
                            translation=pcd_trans)
                    elif t == 'HF':
                        # Undo augmentation
                        gt_boxes.flip('horizontal')
                        # Record augmentation
                        horizontal_flip = np.identity(4)
                        horizontal_flip[1,1] = -1
                        aug_transform = aug_transform @ horizontal_flip
                    elif t == 'VF':
                        # Undo augmentation
                        gt_boxes.flip('vertical')
                        # Record augmentation
                        vertical_flip = np.identity(4)
                        vertical_flip[0,0] = -1
                        aug_transform = aug_transform @ vertical_flip
                    else:
                        raise ValueError(f'unknown transformation "{t}"')

            obj_poses = dict()
            if 'obj_ids' in frame_info:
                obj_ids = [id for id in frame_info['obj_ids']]
                if 'dbsample_group_ids' in input_dict:
                    obj_ids += [self.gid2oid[group_id]
                        for group_id in input_dict['dbsample_group_ids']]
                assert(len(obj_ids) == len(gt_boxes))
                for obj_id, box in zip(obj_ids, gt_boxes.tensor):
                    # Convert bounding box to pose matrix
                    rotmat = R.from_euler('z', -box[6]).as_matrix()
                    # rotmat = R.from_euler('z', -box[-1]-np.pi/2).as_matrix()
                    obj_poses[obj_id] = frame_info['ego_pose'] @ affine_transform(
                        matrix=rotmat, translation=box[:3])
        else:
            obj_poses = frame_info['obj_poses']

        frame_info = dict(
            ego_pose=frame_info['ego_pose'] @ np.linalg.inv(aug_transform),
            scene_id=frame_info['scene_id'], 
            obj_poses=obj_poses)
        return frame_info

    def __call__(self, input_dict):
        sample_idx = input_dict['sample_idx']
        frame_info = self.get_frame_info(input_dict)
        points = self.loader.load(frame_info)

        # Populate frame index in the sequence
        sample_idx = self.loader.frame_id_map.get(sample_idx, sample_idx)
        frame_info = self.loader.frame_infos[sample_idx]
        scene_info = self.loader.scene_infos[frame_info['scene_id']]
        input_dict['frame_idx'] = scene_info['frame_ids'].index(sample_idx)

        if self.use_dims is not None:
            points = points[:,self.use_dims]

        for load_as in self.load_as:
            if load_as not in input_dict or self.replace_existing:
                if self.load_format == 'numpy':
                    input_dict[load_as] = points
                elif self.load_format == 'tensor':
                    input_dict[load_as] = torch.as_tensor(
                        points, dtype=torch.float32, device=torch.device('cpu'))
                elif self.load_format == 'mmdet3d':
                    input_dict[load_as] = LiDARPoints(points, points_dim=len(self.use_dims))
                else:
                    raise ValueError(f'unknown format {self.load_format}')
            else:
                existing = input_dict[load_as]
                if self.use_dims is not None:
                    existing = existing[:,:len(self.use_dims)]
                if isinstance(existing, np.ndarray):
                    input_dict[load_as] = np.concatenate([existing, points])
                elif isinstance(existing, torch.Tensor):
                    input_dict[load_as] = torch.cat([existing, torch.as_tensor(
                        points, dtype=existing.dtype, device=existing.device)])
                elif isinstance(existing, LiDARPoints):
                    input_dict[load_as] = LiDARPoints.cat([existing,
                        LiDARPoints(points, points_dim=len(self.use_dims))])
                else:
                    raise ValueError(f'unknown format {type(existing)}')

        ########################################################################
        # Debug visualization
        ########################################################################
        # existing = existing.tensor.cpu().numpy()
        #
        # import matplotlib.pyplot as plt
        # def plot_boxes(ax, boxes, labels=None, scores=None, cmap=['tab:red']):
        #     from matplotlib.transforms import Affine2D
        #     if labels is None:
        #         labels = [0] * len(boxes)
        #     if scores is None:
        #         scores = [1] * len(boxes)
        #     for box, label, score in zip(boxes, labels, scores):
        #         transform = Affine2D().rotate_around(*box[:2], -box[6]) + ax.transData
        #         rect = plt.Rectangle(box[:2]-box[3:5]/2, width=box[3], height=box[4],
        #                              fc=cmap[label], alpha=float(score)*0.25, transform=transform)
        #         ax.add_patch(rect)
        #         rect = plt.Rectangle(box[:2]-box[3:5]/2, width=box[3], height=box[4],
        #                             ec=cmap[label], alpha=float(score), fill=False, transform=transform)
        #         ax.add_patch(rect)
        # plt.figure(figsize=(12*3, 12))
        # plt.subplot(1, 3, 1)
        # plt.scatter(existing[:,0], existing[:,1], s=1, c='tab:blue')
        # plot_boxes(plt.gca(), input_dict['gt_bboxes_3d'].tensor)
        # plt.title('Sparse')
        # plt.xlim((-25, 25))
        # plt.ylim((-25, 25))
        # plt.axis('off')
        # plt.tight_layout()
        # plt.subplot(1, 3, 2)
        # plt.scatter(points[:,0], points[:,1], s=1, c='tab:orange')
        # plot_boxes(plt.gca(), input_dict['gt_bboxes_3d'].tensor)
        # plt.title('Complete')
        # plt.xlim((-25, 25))
        # plt.ylim((-25, 25))
        # plt.axis('off')
        # plt.tight_layout()
        # plt.subplot(1, 3, 3)
        # plt.scatter(existing[:,0], existing[:,1], s=1, c='tab:blue')
        # plt.scatter(points[:,0], points[:,1], s=1, c='tab:orange')
        # plot_boxes(plt.gca(), input_dict['gt_bboxes_3d'].tensor)
        # plt.title('Combined')
        # plt.xlim((-25, 25))
        # plt.ylim((-25, 25))
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('vis.png')
        #
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:,:3]))
        # colors = np.zeros((points.shape[0], 3))
        # colors[:,0] = np.log(points[:,3]+1) / np.log(256)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # est = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(existing[:,:3]))
        # colors = np.zeros((existing.shape[0], 3))
        # colors[:,2] = np.log(existing[:,3]+1) / np.log(256)
        # est.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd, est])
        # raise
        ########################################################################

        return input_dict
