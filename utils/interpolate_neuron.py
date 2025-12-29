import numpy as np
from scipy.spatial import KDTree

def _find_neighbor(neuron_pt_tuple, t_miss, n_miss, K=5):
        # mask nan values
        nan_mask = np.isnan(neuron_pt_tuple[:, :, [0,1,2]]).any(axis=2)
        nan_indices = np.argwhere(nan_mask)

        missing_counts = nan_mask.sum(axis=0)
        stable_neurons = np.where(missing_counts == 0)[0]

        valid_t_indices = np.where(nan_mask[:, n_miss] == False)[0]
        distances_to_t_miss = np.abs(valid_t_indices - t_miss)
        t_ref = valid_t_indices[np.argmin(distances_to_t_miss)]

        # build kdtree
        coords_ref = neuron_pt_tuple[t_ref, :, 0:3]
        target_coord_ref = coords_ref[n_miss]
        from scipy.spatial import KDTree
        # clean up nan values in coords_ref
        valid_mask_ref = ~np.isnan(coords_ref).any(axis=1)
        coords_ref_clean = coords_ref[valid_mask_ref]
        original_indices_mapping = np.where(valid_mask_ref)[0]
        kdtree = KDTree(coords_ref_clean)
        distances, clean_indices = kdtree.query(target_coord_ref, k=K+1)
        original_neighbor_indices = original_indices_mapping[clean_indices]

        if original_neighbor_indices[0] == n_miss:
                potential_neighbor_indices = original_neighbor_indices[1:]
                potential_distances = distances[1:]

                is_present_at_t_miss = (nan_mask[t_miss, potential_neighbor_indices] == False)
                stable_neighbor_indices = potential_neighbor_indices[is_present_at_t_miss]
                if len(stable_neighbor_indices) == 0 and K < 20:
                        print(f"No stable neighbors found for neuron {n_miss} at time {t_miss} with K={K}")
                        stable_neighbor_indices = _find_neighbor(neuron_pt_tuple, t_miss, n_miss, K=K+5)

        return stable_neighbor_indices, t_ref

def _interpolate_features(neuron_pt_tuple, t_miss, n_miss, t_ref, stable_neighbor_indices):

    target_coord_ref = neuron_pt_tuple[t_ref, n_miss, 0:3]
    coords_neighbors_ref = neuron_pt_tuple[t_ref, stable_neighbor_indices, 0:3]
    coords_neighbors_miss = neuron_pt_tuple[t_miss, stable_neighbor_indices, 0:3]
    # compute transformation vectors from ref to miss for neighbors
    displacement_vectors = coords_neighbors_miss - coords_neighbors_ref
    avg_displacement = np.nanmean(displacement_vectors, axis=0)
    estimated_coord = target_coord_ref + avg_displacement

    features_ref = neuron_pt_tuple[t_ref, n_miss, 3:]
    estamated_features = features_ref  # assuming features do not change significantly
    estimated_full_tuple = np.concatenate([estimated_coord, estamated_features], axis=0)
    return estimated_full_tuple

def interpolate_missing_neurons(neuron_pt_tuple, K=5):
    neuron_pt_tuple_filled = np.copy(neuron_pt_tuple)

    nan_mask_original = np.isnan(neuron_pt_tuple[:, :, [0,1,2]]).any(axis=2)
    nan_indices = np.argwhere(nan_mask_original)

    fill_count = 0
    fail_count = 0
    for t_miss, n_miss in nan_indices:
        stable_neighbor_indices, t_ref = _find_neighbor(neuron_pt_tuple, t_miss, n_miss, K=K)
        if len(stable_neighbor_indices) > 0:
            estimated_full_tuple = _interpolate_features(neuron_pt_tuple, t_miss, n_miss, t_ref, stable_neighbor_indices)
            neuron_pt_tuple_filled[t_miss, n_miss, :] = estimated_full_tuple
            fill_count += 1
        else:
            fail_count += 1
    
    print(f"Filled {fill_count} missing neurons, failed to fill {fail_count} missing neurons.")
    return neuron_pt_tuple_filled

if __name__ == "__main__":
    neuron_pt_tuple_path = r"Z:\data4\WJH\olfactory_result\20251026\test\deg_t45ww0.6\all_neuron_pt_tuple.npy"
    neuron_pt_tuple = np.load(neuron_pt_tuple_path)
    neuron_pt_tuple_filled = interpolate_missing_neurons(neuron_pt_tuple, K=5)