import numpy as np
from scipy.spatial import KDTree
from ..utils.transforms import rotate2d, apply_transform
from ..utils.o3dutils import (
    voxel_grid_downsample,
    statistical_outlier_removal,
    estimate_normals,
)


class PointCloudAggregator():
    def __init__(self,
                 align=True,
                 combine=True,
                 mirror=True,
                 mirror_axis=0,
                 propagate_labels=True,
                 keep_labels=None,
                 down_sample=True,
                 ds_voxel_size=0.02,
                 ds_average=False,
                 remove_outliers=True,
                 sor_knn=200,
                 sor_std_ratio=2.0,
                 estimate_normals=True,
                 ne_radius=0.1,
                 ne_knn=500):
        self.align = align
        self.combine = combine
        self.mirror = mirror
        self.mirror_axis = mirror_axis
        self.propagate_labels = propagate_labels
        self.keep_labels = keep_labels
        self.down_sample = down_sample
        self.ds_voxel_size = ds_voxel_size
        self.ds_average = ds_average
        self.remove_outliers = remove_outliers
        self.sor_knn = sor_knn
        self.sor_std_ratio = sor_std_ratio
        self.estimate_normals = estimate_normals
        self.ne_radius = ne_radius
        self.ne_knn = ne_knn

    def align_points_with_boxes(self, boxes3d, pts_xyz, pts_dir=None):
        for i in range(len(boxes3d)):
            if boxes3d[i] is None:
                continue

            pts_xyz[i] -= boxes3d[i][:3]
            pts_xyz[i][:, :2] = rotate2d(
                pts_xyz[i][:, :2], boxes3d[i][6])
            if pts_dir is not None:
                pts_dir[i][:, :2] = rotate2d(
                    pts_dir[i][:, :2], boxes3d[i][6])

        if pts_dir is None:
            return dict(pts_xyz=pts_xyz)
        else:
            return dict(pts_xyz=pts_xyz, pts_dir=pts_dir)

    def align_points_with_pose(self, ego_pose, pts_xyz, pts_dir=None):
        for i in range(len(ego_pose)):
            if ego_pose[i] is None:
                continue

            pts_xyz[i] = apply_transform(ego_pose[i], pts_xyz[i])
            if pts_dir is not None:
                pts_dir[i] = np.dot(pts_dir[i], ego_pose[i][:3, :3].T)

        if pts_dir is None:
            return dict(pts_xyz=pts_xyz)
        else:
            return dict(pts_xyz=pts_xyz, pts_dir=pts_dir)

    def align_points(self, pts_xyz, pts_dir=None, boxes3d=None, ego_pose=None, **kwargs):
        if boxes3d is not None and ego_pose is not None:
            raise ValueError('only one of boxes3d and ego_pose can be specified')
        if boxes3d is not None:
            return self.align_points_with_boxes(boxes3d, pts_xyz, pts_dir)
        if ego_pose is not None:
            return self.align_points_with_pose(ego_pose, pts_xyz, pts_dir)

    def combine_points(self, pts_xyz, pts_fields, **kwargs):
        def is_not_none(x):
            return x is not None
        pts_xyz = np.concatenate(list(filter(is_not_none, pts_xyz)))

        results = dict(pts_xyz=pts_xyz)
        for k in pts_fields:
            results[k] = np.concatenate(list(filter(is_not_none, kwargs[k])))
        return results

    def mirror_points(self, pts_xyz, pts_dir, pts_fields, **kwargs):
        pts_xyz_ = pts_xyz.copy()
        pts_xyz_[:, self.mirror_axis] *= -1
        results = dict(pts_xyz=np.concatenate([pts_xyz, pts_xyz_]))

        if pts_dir is not None:
            pts_dir_ = pts_dir.copy()
            pts_dir_[:, self.mirror_axis] *= -1
            results['pts_dir'] = np.concatenate([pts_dir, pts_dir_])

        for k in pts_fields:
            if k == 'pts_dir':
                continue
            results[k] = np.tile(kwargs[k], (2, 1))
        return results

    def process_pts_labels(self, boxes3d, pts_xyz, pts_fields, **kwargs):
        results = dict()

        pts_labels = kwargs.get('pts_labels', None)
        if self.propagate_labels:
            lab_mask = -1 if pts_labels is None else np.all(pts_labels >= 0, axis=-1)
            if np.sum(lab_mask) > 0:
                lab_pts_xyz = pts_xyz[lab_mask]
                lab_pts_labels = pts_labels[lab_mask]

                # Propagate semantic labels with nearest neighbors
                _, ind = KDTree(lab_pts_xyz).query(pts_xyz[~lab_mask], workers=-1)
                pts_labels[~lab_mask] = lab_pts_labels[ind]
                results['pts_labels'] = pts_labels

        if self.keep_labels is not None:
            results['rej_pts_xyz'] = rej_pts_xyz = dict()
            results['rej_pts_dir'] = rej_pts_dir = dict()
            rej_pts_fields = []
            for k in pts_fields:
                k = 'rej_' + k
                rej_pts_fields.append(k)
                results[k] = dict()

            # Identify object points based on semantic labels
            keep_mask = np.zeros(pts_xyz.shape[0], dtype=bool)
            for label in self.keep_labels:
                keep_mask |= np.all(pts_labels == label, axis=-1)

            # Isolate and restore background points
            if boxes3d is not None and 'pts_feats' in kwargs:
                rej_mask = ~keep_mask
                rej_pts_idx = kwargs['pts_feats'][rej_mask][:, -1].astype(int)
                for i in set(rej_pts_idx.tolist()):
                    frame_mask = rej_pts_idx == i
                    rej_pts_xyz[i] = pts_xyz[rej_mask][frame_mask]
                    rej_pts_xyz[i][:, :2] = rotate2d(
                        rej_pts_xyz[i][:, :2], -boxes3d[i][6])
                    rej_pts_xyz[i] += boxes3d[i][:3]
                    rej_pts_dir[i] = kwargs['pts_dir'][rej_mask][frame_mask]
                    rej_pts_dir[i][:, :2] = rotate2d(
                        rej_pts_dir[i][:, :2], -boxes3d[i][6])
                    for k, l in zip(rej_pts_fields, pts_fields):
                        results[k][i] = kwargs[l][rej_mask][frame_mask]

            results['pts_xyz'] = pts_xyz[keep_mask]
            for k in pts_fields:
                results[k] = kwargs[k][keep_mask]

        return results

    def voxel_grid_downsample(self, pts_xyz, pts_fields, **kwargs):
        if pts_xyz.shape[0] > self.sor_knn:
            if not self.ds_average:
                _, ind = voxel_grid_downsample(
                    pts_xyz, self.ds_voxel_size, average=False)
                results = dict(pts_xyz=pts_xyz[ind])
                for k in pts_fields:
                    results[k] = kwargs[k][ind]
            else:
                results = dict()
                for k in pts_fields:
                    pts = np.concatenate([pts_xyz, kwargs[k]], axis=1)
                    pts = voxel_grid_downsample(
                        pts, self.ds_voxel_size, average=True)
                    results['pts_xyz'] = pts[:,:3]
                    results[k] = pts[:,3:]
        else:
            results = dict(pts_xyz=pts_xyz)
            for k in pts_fields:
                results[k] = kwargs[k]
        return results

    def statistical_outlier_removal(self, pts_xyz, pts_fields, **kwargs):
        if pts_xyz.shape[0] >= self.sor_knn:
            _, ind = statistical_outlier_removal(
                pts_xyz, self.sor_knn, self.sor_std_ratio, return_indices=True)
            results = dict(pts_xyz=pts_xyz[ind])
            for k in pts_fields:
                results[k] = kwargs[k][ind]
        else:
            results = dict(pts_xyz=pts_xyz)
            for k in pts_fields:
                results[k] = kwargs[k]
        return results

    def hybrid_normal_estimation(self, pts_xyz, **kwargs):
        if pts_xyz.shape[0] >= 3:
            return dict(pts_normal=estimate_normals(
                pts_xyz, radius=self.ne_radius, k=self.ne_knn).astype(np.float32))
        else:
            return dict(pts_normal=np.zeros_like(pts_xyz))

    def compuate_incidence(self, pts_normal, pts_dir, **kwargs):
        pts_incidence = np.arccos(
            pts_normal[:, 0] * pts_dir[:, 0] +
            pts_normal[:, 1] * pts_dir[:, 1] +
            pts_normal[:, 2] * pts_dir[:, 2])
        flip_mask = pts_incidence > np.pi/2
        pts_normal[flip_mask] *= -1
        pts_incidence[flip_mask] = np.pi - pts_incidence[flip_mask]
        return dict(pts_incidence=pts_incidence)

    def __call__(self, pts_xyz, boxes3d=None, ego_pose=None, **kwargs):
        results = dict(boxes3d=boxes3d, ego_pose=ego_pose,
                       pts_xyz=pts_xyz, pts_fields=[])

        for k, v in kwargs.items():
            if k.startswith('pts_'):
                results['pts_fields'].append(k)
                results[k] = v

        if self.align:
            results.update(self.align_points(**results))
        if self.combine:
            results.update(self.combine_points(**results))
        if self.mirror:
            results.update(self.mirror_points(**results))
        results.update(self.process_pts_labels(**results))
        if self.down_sample:
            results.update(self.voxel_grid_downsample(**results))
        if self.remove_outliers:
            results.update(self.statistical_outlier_removal(**results))
        if self.estimate_normals:
            results.update(self.hybrid_normal_estimation(**results))
            results.update(self.compuate_incidence(**results))

        del results['boxes3d']
        del results['ego_pose']
        del results['pts_fields']

        return results


class ObjectAggregator(PointCloudAggregator):
    pass


class SceneAggregator(PointCloudAggregator):
    def __init__(self, **kwargs):
        default_kwargs = dict(mirror=False)
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
