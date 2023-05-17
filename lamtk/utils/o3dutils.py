import numpy as np
import torch
import open3d as o3d

# Expose the original package
o3d = o3d

def PointCloud(points):
    if isinstance(points, o3d.geometry.PointCloud):
        return points
    else:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:,:3]))
        # Store additional features as colors
        # 0 <= num_features <= 3
        num_features = min(max(points.shape[1] - 3, 0), 3)
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate([
            points[:,3:3+num_features],
            np.zeros((points.shape[0], 3-num_features))
        ], axis=-1))
        return pcd


def to_numpy(pcd, use_dim=6):
    if isinstance(pcd, np.ndarray):
        return pcd
    else:
        return np.concatenate(
            [pcd.points, pcd.colors], axis=-1
        ).astype(np.float32)[:,:use_dim]


def radius_outlier_removal(points, nb_points, radius, print_progress=False, return_indices=False):
    pcd = PointCloud(points)
    pcd, ind = pcd.remove_radius_outlier(nb_points, radius, print_progress)
    if isinstance(points, o3d.geometry.PointCloud):
        return (pcd, ind) if return_indices else pcd
    else:
        return (points[ind], ind) if return_indices else points[ind]
ror = radius_outlier_removal


def statistical_outlier_removal(points, nb_neighbors, std_ratio, print_progress=False, return_indices=False):
    pcd = PointCloud(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio, print_progress)
    if isinstance(points, o3d.geometry.PointCloud):
        return (pcd, ind) if return_indices else pcd
    else:
        return (points[ind], ind) if return_indices else points[ind]
sor = statistical_outlier_removal


def _voxel_grid_downsample_with_average(points, voxel_size=0.01):
    pcd = PointCloud(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    if isinstance(points, o3d.geometry.PointCloud):
        return pcd
    else:
        return to_numpy(pcd, points.shape[1])


def _voxel_grid_downsample_no_average(points, voxel_size=0.01):
    import open3d.ml.torch as ml3d

    indices = np.arange(points.shape[0])
    np.random.shuffle(indices)
    points_tensor = torch.tensor(points[indices], dtype=torch.float32)
    result = ml3d.ops.voxelize(
        points_tensor,
        torch.tensor([0,points.shape[0]]),
        torch.tensor([voxel_size, voxel_size, voxel_size], dtype=torch.float32),
        points_tensor.min(0).values,
        points_tensor.max(0).values)
    voxel_indices = result.voxel_point_indices[result.voxel_point_row_splits[:-1]].numpy()
    return points[indices[voxel_indices]], indices[voxel_indices]


def voxel_grid_downsample(points, voxel_size=0.01, average=True):
    if average:
        return _voxel_grid_downsample_with_average(points, voxel_size)
    else:
        return _voxel_grid_downsample_no_average(points, voxel_size)


def estimate_normals(points, radius=0, k=30, orient=None, reference=None):
    if (radius is not None and radius != 0) and (k is not None and k != 0):
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius, k)
    elif radius is not None and radius != 0:
        search_param = o3d.geometry.KDTreeSearchParamRadius(radius)
    elif k is not None and k != 0:
        search_param = o3d.geometry.KDTreeSearchParamKNN(k)
    else:
        raise ValueError('Neither \'radius\' or \'k\' is specified.')
    pcd = PointCloud(points)
    pcd.estimate_normals(search_param)
    if orient == 'consistent' or orient == 'consistency':
        if reference is None:
            reference = k
        pcd.orient_normals_consistent_tangent_plane(reference)
    elif orient == 'direction':
        if reference is None:
            reference = [0.0, 0.0, 1.0]
        pcd.orient_normals_to_align_with_direction(reference)
    elif orient == 'camera':
        if reference is None:
            reference = [0.0, 0.0, 0.0]
        pcd.orient_normals_towards_camera_location(reference)
    return np.asarray(pcd.normals)


_icp_requires_normals = dict(
    point_to_point=(False, False),
    point_to_plane=(False, True),
    generalized_icp=(True, True),
    colored_icp=(False, True)
)

def iterative_closest_point(
    source, target, 
    max_correspondence_distance,
    init=np.identity(4),
    method='point_to_point',
    max_iter=30,
    voxel_radius=None,
    normal_estimation_param=None
):
    ############################################################################
    # Check arguments
    ############################################################################
    source = PointCloud(source)
    target = PointCloud(target)

    if method == 'point_to_point':
        registration_icp = o3d.pipelines.registration.registration_icp
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif method == 'point_to_plane':
        registration_icp = o3d.pipelines.registration.registration_icp
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif method == 'generalized_icp':
        registration_icp = o3d.pipelines.registration.registration_generalized_icp
        estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    elif method == 'colored_icp':
        registration_icp = o3d.pipelines.registration.registration_colored_icp
        estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP()

    if isinstance(max_correspondence_distance, (tuple, list, np.ndarray)):
        num_scales = len(max_correspondence_distance)
    else:
        max_correspondence_distance = [max_correspondence_distance]
        num_scales = 1

    if isinstance(max_iter, (tuple, list, np.ndarray)):
        if len(max_iter) != num_scales:
            # TODO: add error message
            raise ValueError()
    else:
        max_iter = [max_iter] * num_scales

    if voxel_radius is None:
        voxel_radius = [None] * num_scales
    elif isinstance(voxel_radius, (tuple, list, np.ndarray)):
        if len(voxel_radius) != num_scales:
            # TODO: add error message
            raise ValueError()
    else:
        voxel_radius = [voxel_radius] * num_scales

    if normal_estimation_param is None:
        normal_estimation_param = [dict(k=30)] * num_scales
    elif isinstance(normal_estimation_param, (tuple, list, np.ndarray)):
        if len(normal_estimation_param) != num_scales:
            # TODO: add error message
            raise ValueError()
    else:
        normal_estimation_param = [normal_estimation_param] * num_scales


    ############################################################################
    # Iterative ICP
    ############################################################################
    current_transformation = init
    for scale in range(num_scales):
        if voxel_radius[scale] is not None:
            source_down = source.voxel_down_sample(voxel_radius[scale])
            target_down = target.voxel_down_sample(voxel_radius[scale])
        else:
            source_down = source
            target_down = target

        if _icp_requires_normals[method][0] and not source_down.has_normals():
            estimate_normals(source_down, **normal_estimation_param[scale])

        if _icp_requires_normals[method][1] and not target_down.has_normals():
            estimate_normals(target_down, **normal_estimation_param[scale])

        try:
            result = registration_icp(
                source_down, target_down, max_correspondence_distance[scale],
                current_transformation, estimation_method,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iter[scale]
                )
            )
            current_transformation = result.transformation
        except:
            # ICP failed
            current_transformation = init
    return current_transformation
icp = iterative_closest_point


def incremental_icp(
    point_clouds,
    max_correspondence_distance,
    init=np.identity(4),
    method='point_to_point',
    max_iter=30,
    voxel_radius=None,
    normal_estimation_param=None,
    mask=None,
    num_target_frames=1
):
    if len(point_clouds) == 0:
        return []
    if len(point_clouds) == 1:
        return [np.identity(4)]

    if mask is None:
        mask = [True] * len(point_clouds)
    elif not isinstance(mask, (tuple, list, np.ndarray)) or \
            len(mask) != len(point_clouds):
        raise ValueError()
    mask[0] = True

    # Convert to O3D point cloud format
    pcds = [ PointCloud(point_cloud) for point_cloud in point_clouds ]

    # Prepare the point clouds by skipping and aggregating
    # point clouds with mask==False
    pcds_tmp = []
    for pcd, should_align in zip(pcds, mask):
        if should_align:
            pcds_tmp.append(o3d.geometry.PointCloud())
        pcds_tmp[-1] += pcd
    pcds = pcds_tmp

    transformations = [np.identity(4)]
    global_transformation = np.identity(4)
    for i in range(1, len(pcds)):
        if mask[i] and i >= num_target_frames:
            source = pcds[i].transform(global_transformation)
            if num_target_frames > 1:
                target = o3d.geometry.PointCloud()
                for j in range(1, num_target_frames+1):
                    target += pcds[i-j]
            else:
                target = pcds[i-1]
            pair_transformation = icp(
                source, target,
                max_correspondence_distance, init, method,
                max_iter, voxel_radius, normal_estimation_param
            )
            pcds[i] = source.transform(pair_transformation)
            global_transformation = pair_transformation @ global_transformation
        transformations.append(global_transformation.copy())

    # Map the transformations for the processed point clouds
    # back to the original point clouds
    i = 0
    transformations_final = [np.identity(4)]
    for should_align in mask[1:]:
        if should_align:
            i += 1
        transformations_final.append(transformations[i])

    return transformations_final


def poisson_reconstruction(points, normals, alpha=0.01, *args, **kwargs):
    pcd = PointCloud(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, *args, **kwargs
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, alpha))

    return mesh
