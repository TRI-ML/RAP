import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import math
import os

import fire
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm

from waymo_open_dataset.protos import scenario_pb2


def scalar_to_one_hot(length, index, has_zero=False):
    if has_zero:
        offset = 1
    else:
        offset = 0
    assert 0 <= index < length + offset
    if index + 1 - offset > 0:
        one_hot_type = np.eye(length)[index - offset]
    else:
        one_hot_type = np.zeros(length)

    return one_hot_type


def group_tracks(tracks):
    object_types = {
        "TYPE_UNSET": 0,
        "TYPE_VEHICLE": 1,
        "TYPE_PEDESTRIAN": 2,
        "TYPE_CYCLIST": 3,
        "TYPE_OTHER": 4,
    }
    state_size = 11
    traj = np.zeros((len(tracks), len(tracks[0].states), state_size))
    mask_traj = np.zeros((len(tracks), len(tracks[0].states)), dtype=bool)
    traj_type = np.zeros((len(tracks), len(object_types) - 1))
    id_to_idx = {}

    for i_track, track in enumerate(tracks):
        traj_type[i_track, :] = scalar_to_one_hot(
            len(object_types) - 1, track.object_type, has_zero=True
        )
        id_to_idx[track.id] = i_track
        for i_time, state in enumerate(track.states):
            if state.valid:
                traj[i_track, i_time, 0] = state.center_x
                traj[i_track, i_time, 1] = state.center_y
                traj[i_track, i_time, 2] = state.heading
                traj[i_track, i_time, 3] = state.velocity_x
                traj[i_track, i_time, 4] = state.velocity_y
                traj[i_track, i_time, 5] = state.width
                traj[i_track, i_time, 6] = state.length
                traj[i_track, i_time, 7:11] = traj_type[i_track, :]
                mask_traj[i_track, i_time] = state.center_x != 0 or state.center_y != 0
            else:
                mask_traj[i_track, i_time] = False

    # Remove trajectories that are masked for the whole time
    mask_any_time = mask_traj.any(-1)
    to_delete = []
    for key, value in id_to_idx.items():
        if not mask_any_time[value]:
            to_delete.append(key)
        else:
            id_to_idx[key] = np.sum(mask_any_time[:value])
    for key in to_delete:
        del id_to_idx[key]
    traj = traj[mask_any_time]
    traj_type = traj_type[mask_any_time]
    mask_traj = mask_traj[mask_any_time]
    # traj:(n_agents, seq_time, features), mask:(n_agents, seq_time), traj_type:(n_agents, features)
    assert (traj[..., :2][mask_traj] != 0).any(-1).all()
    return traj, mask_traj, traj_type, id_to_idx


def filter_tracks(
    pos,
    trajs,
    mask_trajs,
    trajs_type,
    to_predict,
    id_to_idx,
    mask_keep,
    max_moving_distance,
    max_static_distance,
):
    distances2 = ((trajs[:, :, :2] - pos[None, None, :]) ** 2).sum(-1).min(1)
    first_non_0_pos = np.take_along_axis(
        trajs, np.argmax(mask_trajs, 1)[:, None, None], axis=1
    )
    is_moving = (
        np.abs((trajs[:, :, :2] - first_non_0_pos[:, 0:1, :2]) * mask_trajs[:, :, None])
        .sum(1)
        .sum(1)
        > 1
    )
    filtered = np.zeros_like(distances2, dtype=bool)
    filtered[is_moving] = distances2[is_moving] < max_moving_distance**2
    filtered[np.logical_not(is_moving)] = (
        distances2[np.logical_not(is_moving)] < max_static_distance**2
    )
    filtered = np.logical_or(filtered, mask_keep)

    # Filter out trajectories
    to_delete = []
    idx_to_id = {}
    for key, value in id_to_idx.items():
        if not filtered[value]:
            to_delete.append(key)
        else:
            new_value = np.sum(filtered[:value])
            idx_to_id[new_value] = key
            id_to_idx[key] = new_value
    for key in to_delete:
        del id_to_idx[key]

    trajs = trajs[filtered]
    trajs_type = trajs_type[filtered]
    mask_trajs = mask_trajs[filtered]
    to_predict = to_predict[filtered]

    if mask_keep.all():
        return trajs, mask_trajs, trajs_type, to_predict, id_to_idx

    # Sort entries from closest to furthest to input pos
    distances2 = distances2[filtered]
    distance_sort = np.argsort(distances2)
    copy_trajs = trajs.copy()
    copy_mask_trajs = mask_trajs.copy()
    copy_trajs_type = trajs_type.copy()
    copy_to_predict = to_predict.copy()
    skip = np.argmin(mask_keep)
    assert skip > 1
    offset = skip
    for i, idx in enumerate(distance_sort[skip:]):
        if idx > skip:
            ii = i + offset
            trajs[ii] = copy_trajs[idx]
            trajs_type[ii] = copy_trajs_type[idx]
            mask_trajs[ii] = copy_mask_trajs[idx]
            to_predict[ii] = copy_to_predict[idx]
            id_to_idx[idx_to_id[idx]] = ii
        else:
            offset -= 1
    assert (trajs[..., :2][mask_trajs] != 0).any(-1).all()
    return trajs, mask_trajs, trajs_type, to_predict, id_to_idx


def cut_lane(lane, pos, max_len):
    center_idx = np.argmin(((lane - pos[None, :]) ** 2).sum(-1))
    start = max(0, center_idx - max_len // 2)
    return lane[start : start + max_len, :]


def group_lanes(map, center, max_lane_len, max_lane_distance):
    all_objects = []
    all_types = []
    max_len = 0
    id_to_idx = {}
    stride = 2
    max_lane_len = max_lane_len * stride
    for object in map:
        # Type one_hot encoding is as follows: 0: lane, 1: stop_sign, 2: cross_walk, 3: speed_bump
        lane = object.lane.polyline
        is_cut_lane = len(lane) > max_lane_len
        len_lane = min(len(lane), max_lane_len)
        len_cross_walk = len(object.crosswalk.polygon)
        len_speed_bump = len(object.speed_bump.polygon)
        num_obj_types = 4

        max_len = max(max_len, len_lane)
        max_len = max(max_len, len_cross_walk)
        max_len = max(max_len, len_speed_bump)
        if len_lane > 0:
            current_lane = np.zeros((len(lane), 2))
            for i_point, cw in enumerate(lane):
                current_lane[i_point, 0] = cw.x
                current_lane[i_point, 1] = cw.y
            if is_cut_lane:
                current_lane = cut_lane(current_lane, center, max_lane_len)
            min_distance2 = np.min(((current_lane - center[None, :]) ** 2).sum(-1))
            if min_distance2 < max_lane_distance**2:
                id_to_idx[object.id] = len(all_objects)
                all_objects.append(current_lane)
                all_types.append(scalar_to_one_hot(num_obj_types, 0))
        # elif len_cross_walk > 0:
        #     current_cross_walk = np.zeros((len_cross_walk, 2))
        #     for i_point, cw in enumerate(object.crosswalk.polygon):
        #         current_cross_walk[i_point, 0] = cw.x
        #         current_cross_walk[i_point, 1] = cw.y
        #     all_objects.append(current_cross_walk)
        #     all_types.append(scalar_to_one_hot(num_obj_types, 2))
        # elif len_speed_bump > 0:
        #     current_speed_bump = np.zeros((len_speed_bump, 2))
        #     for i_point, cw in enumerate(object.speed_bump.polygon):
        #         current_speed_bump[i_point, 0] = cw.x
        #         current_speed_bump[i_point, 1] = cw.y
        #     all_objects.append(current_speed_bump)
        #     all_types.append(scalar_to_one_hot(num_obj_types, 3))
        # elif not (object.stop_sign.position.x == 0 and object.stop_sign.position.y == 0):
        #     all_objects.append([np.array([object.stop_sign.position.x, object.stop_sign.position.y])])
        #     all_types.append(scalar_to_one_hot(num_obj_types, 1))

    object_array = np.zeros((len(all_objects), (max_len + 1) // stride, 2))
    mask_object_array = np.zeros(
        (len(all_objects), (max_len + 1) // stride), dtype=bool
    )
    object_types_array = np.zeros((len(all_types), num_obj_types))

    for i_object, object in enumerate(all_objects):
        len_object = (len(object) + 1) // stride
        object_array[i_object, :len_object, :] = object[::2]
        mask_object_array[i_object, :len_object] = True
        object_types_array[i_object] = all_types[i_object]
    # for i, lane in enumerate(object_array):
    #     plt.plot(lane[mask_object_array[i, :], 0], lane[mask_object_array[i, :], 1], alpha=0.3)

    idx_to_id = {value: key for key, value in id_to_idx.items()}
    # Sort entries from closest to furthest to input center
    distances2 = np.min(((object_array - center[None, None, :]) ** 2).sum(-1), 1)
    distance_sort = np.argsort(distances2)
    copy_object = object_array.copy()
    copy_mask_object = mask_object_array.copy()
    copy_type = object_types_array.copy()
    for i, idx in enumerate(distance_sort):
        object_array[i] = copy_object[idx]
        mask_object_array[i] = copy_mask_object[idx]
        object_types_array[i] = copy_type[idx]
        id_to_idx[idx_to_id[idx]] = i

    return object_array, mask_object_array, object_types_array, id_to_idx


def group_light_signals(light_signals, id_to_idx, n_map_objects):
    state_to_idx = {
        "TRAFFIC_LIGHT_STATE_UNKNOWN": 0,
        "TRAFFIC_LIGHT_STATE_ARROW_STOP": 1,
        "TRAFFIC_LIGHT_STATE_ARROW_CAUTION": 2,
        "TRAFFIC_LIGHT_STATE_ARROW_GO": 3,
        "TRAFFIC_LIGHT_STATE_STOP": 4,
        "TRAFFIC_LIGHT_STATE_CAUTION": 5,
        "TRAFFIC_LIGHT_STATE_GO": 6,
        "TRAFFIC_LIGHT_STATE_FLASHING_STOP": 7,
        "TRAFFIC_LIGHT_STATE_FLASHING_CAUTION": 8,
    }
    len_time = len(light_signals)
    all_lanes_states = np.zeros((n_map_objects, len_time, len(state_to_idx) - 1))
    for t, lanes_states in enumerate(light_signals):
        for lane in lanes_states.lane_states:
            if lane.lane in id_to_idx.keys():
                all_lanes_states[id_to_idx[lane.lane], t, :] = scalar_to_one_hot(
                    len(state_to_idx) - 1, lane.state, True
                )

    # (n_objects, seq_time, features)
    return all_lanes_states


def normalize_all(traj, map, pos, angle):

    c = math.cos(angle)
    s = math.sin(angle)
    rotation_mat = np.array([[c, s], [-s, c]])
    traj_clone = traj.clone()
    traj_clone[..., :2] = (
        traj_clone[..., :2] - pos.reshape(([1] * (traj.ndim - 1)) + [2])
    ) @ rotation_mat
    traj_clone[..., 2] = (traj_clone[..., 2] + angle + np.pi) % (2 * np.pi) - np.pi
    if traj.shape[-1] >= 5:
        traj_clone[..., 3:5] = traj_clone[..., 3:5] @ rotation_mat
    map_clone = (map.clone() - pos.reshape(([1] * (map.ndim - 1)) + [2])) @ rotation_mat

    return traj_clone, map_clone


def fill_gaps(trajs, mask_in):
    """
    If trajectories are partially observed with gaps (observed then not then observed again), fill the gaps with interpolations.

    Args:

        trajs: size (n_agents, time, features) features are organized as [x, y, angle, vx, vy, other_features ]

    """
    mask = mask_in.copy()
    first_non_zeros = np.argmax(mask, 1)
    last_non_zeros = mask.shape[1] - np.argmax(np.flip(mask, 1), 1)
    has_gaps = np.logical_and(
        last_non_zeros - first_non_zeros > np.maximum(mask.sum(1), 1), mask.sum(1) > 1
    )
    if not has_gaps.any():
        # No gap to fill, returning the input
        return trajs
    # iterate over agents
    for i in range(trajs.shape[0]):
        if has_gaps[i]:
            left = first_non_zeros[i]
            right = first_non_zeros[i]
            for t in range(first_non_zeros[i], last_non_zeros[i]):
                if mask[i, t] and left == right:
                    left += 1
                elif mask[i, t]:
                    break
                else:
                    mask[i, t] = True
                right += 1
            # Linear filling for positions:
            trajs[i, left:right, :2] = (np.arange(right - left) / (right - left))[
                :, None
            ] * (trajs[i, right, :2] - trajs[i, left - 1, :2])[None, :] + trajs[
                i, left - 1 : left, :2
            ]
            # Linear filling for velocities and the rest:
            trajs[i, left:right, 3:] = (np.arange(right - left) / (right - left))[
                :, None
            ] * (trajs[i, right, 3:] - trajs[i, left - 1, 3:])[None, :] + trajs[
                i, left - 1 : left, 3:
            ]
            # Linear filling for angles (periodicity doesn't allow direct interpolation):
            cos_traj = np.cos(trajs[i, left - 1 : right + 1, 2])
            sin_traj = np.sin(trajs[i, left - 1 : right + 1, 2])
            cos_traj = (np.arange(right - left) / (right - left)) * (
                cos_traj[-1] - cos_traj[0]
            ) + cos_traj[0]
            sin_traj = (np.arange(right - left) / (right - left)) * (
                sin_traj[-1] - sin_traj[0]
            ) + sin_traj[0]
            trajs[i, left:right, 2] = np.arctan2(sin_traj, cos_traj)
    # Only the first gap was filled, recursive call to complete others
    return fill_gaps(trajs, mask)


def group_scenario(scenario):
    ids_of_interest = list(set(scenario.objects_of_interest))

    # Only gather scenario with a pair of interacting vehicles
    if len(ids_of_interest) != 2:
        return None

    traj, mask_traj, traj_type, id_to_idx = group_tracks(scenario.tracks)
    assert (traj[..., :2][mask_traj] != 0).any(-1).all()

    to_predict = np.zeros(traj.shape[0], dtype=bool)
    for idx in scenario.tracks_to_predict:
        to_predict[idx.track_index] = True

    # # Set ego as the first agent in the list of trajectories
    # index_ego = scenario.sdc_track_index
    # if index_ego != 0:
    #     for key, value in id_to_idx.items():
    #         if value == 0:
    #             id_0 = key
    #     traj[[0, index_ego]] = traj[[index_ego, 0]]
    #     mask_traj[[0, index_ego]] = mask_traj[[index_ego, 0]]
    #     traj_type[[0, index_ego]] = traj_type[[index_ego, 0]]
    #     to_predict[[0, index_ego]] = to_predict[[index_ego, 0]]
    #     id_to_idx[id_0] = index_ego
    #     id_to_idx[scenario.sdc_track_index] = 0

    # Set the agents of interest as the first agents in the list of trajectories
    for key, value in id_to_idx.items():
        if value == 0:
            id_0 = key
        elif value == 1:
            id_1 = key
    indices_of_interest = sorted(
        [id_to_idx[ids_of_interest[0]], id_to_idx[ids_of_interest[1]]]
    )
    traj[[0, indices_of_interest[0]]] = traj[
        [
            indices_of_interest[0],
            0,
        ]
    ]
    mask_traj[[0, indices_of_interest[0]]] = mask_traj[
        [
            indices_of_interest[0],
            0,
        ]
    ]
    traj_type[[0, indices_of_interest[0]]] = traj_type[
        [
            indices_of_interest[0],
            0,
        ]
    ]
    to_predict[[0, indices_of_interest[0]]] = to_predict[
        [
            indices_of_interest[0],
            0,
        ]
    ]
    traj[[1, indices_of_interest[1]]] = traj[[indices_of_interest[1], 1]]
    mask_traj[[1, indices_of_interest[1]]] = mask_traj[[indices_of_interest[1], 1]]
    traj_type[[1, indices_of_interest[1]]] = traj_type[[indices_of_interest[1], 1]]
    to_predict[[1, indices_of_interest[1]]] = to_predict[[indices_of_interest[1], 1]]

    id_to_idx[id_0] = id_to_idx[ids_of_interest[0]]
    id_to_idx[ids_of_interest[0]] = 0
    id_to_idx[id_1] = id_to_idx[ids_of_interest[1]]
    id_to_idx[ids_of_interest[1]] = 1

    assert (traj[..., :2][mask_traj] != 0).any(-1).all()

    # ego_current_state = scenario.tracks[scenario.sdc_track_index].states[scenario.current_time_index]
    # angle = ego_current_state.heading
    traj = fill_gaps(traj, mask_traj)
    pos = traj[0, scenario.current_time_index, :2]
    angle = traj[0, scenario.current_time_index, 2]
    # mask_agent_of_interest = np.zeros((traj.shape[0]), dtype=bool)
    # idx_of_interest = [id_to_idx[id] for id in scenario.objects_of_interest]
    # mask_agent_of_interest[idx_of_interest] = True

    traj, mask_traj, traj_type, to_predict, id_to_idx = filter_tracks(
        pos,
        traj,
        mask_traj,
        traj_type,
        to_predict,
        id_to_idx,
        mask_keep=to_predict,
        max_moving_distance=50,
        max_static_distance=30,
    )

    assert (traj[..., :2][mask_traj] != 0).any(-1).all()
    if traj.shape[0] > 100:
        print(traj.shape[0])

    map, mask_map, map_type, map_id_to_idx = group_lanes(
        scenario.map_features, pos, max_lane_len=50, max_lane_distance=50
    )

    lane_states = group_light_signals(
        scenario.dynamic_map_states, map_id_to_idx, map.shape[0]
    )

    traj, map = normalize_all(traj, map, pos, -angle)
    assert (
        (
            traj[0, scenario.current_time_index + 1 :, :2][
                mask_traj[0, scenario.current_time_index + 1 :]
            ]
            != 0
        )
        .any(-1)
        .all()
    )
    assert (
        (
            traj[0, : scenario.current_time_index, :2][
                mask_traj[0, : scenario.current_time_index]
            ]
            != 0
        )
        .any(-1)
        .all()
    )
    assert (traj[1:, :, :2][mask_traj[1:, :]] != 0).any(-1).all()

    len_pred = traj.shape[1] - scenario.current_time_index - 1

    traj = traj.transpose((1, 0, 2))
    mask_traj = mask_traj.transpose((1, 0))
    map = map.transpose((1, 0, 2))
    mask_map = mask_map.transpose((1, 0))
    assert (
        (
            traj[scenario.current_time_index + 1 :, 0, :2][
                mask_traj[scenario.current_time_index + 1 :, 0]
            ]
            != 0
        )
        .any(-1)
        .all()
    )
    assert (
        (
            traj[: scenario.current_time_index, 0, :2][
                mask_traj[: scenario.current_time_index, 0]
            ]
            != 0
        )
        .any(-1)
        .all()
    )
    assert (traj[:, 1:, :2][mask_traj[:, 1:]] != 0).any(-1).all()

    # Mask futures for trajectories that are not to be predicted
    traj = traj * mask_traj[:, :, None]

    # to_predict[0] = True
    # to_predict[1] = True
    # mask_traj[scenario.current_time_index+1:, np.logical_not(to_predict)] = 0
    mask_to_predict = mask_traj.copy()
    mask_to_predict[:, np.logical_not(to_predict)] = False
    assert (
        (
            traj[scenario.current_time_index + 1 :, 0, :2][
                mask_to_predict[scenario.current_time_index + 1 :, 0]
            ]
            != 0
        )
        .any(-1)
        .all()
    )
    assert (
        (
            traj[: scenario.current_time_index, 0, :2][
                mask_to_predict[: scenario.current_time_index, 0]
            ]
            != 0
        )
        .any(-1)
        .all()
    )
    assert (traj[:, 1:, :2][mask_to_predict[:, 1:]] != 0).any(-1).all()

    return {
        "traj": traj,
        "mask_traj": mask_traj,
        "mask_to_predict": mask_to_predict,
        "lanes": map,
        "lane_states": lane_states,
        "mask_lanes": mask_map,
        "len_pred": len_pred,
        "mean_pos": pos,
    }


def preprocess_scenario(data, output_dir):
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(data.numpy())
    scenario_id = scenario.scenario_id
    scenario = group_scenario(scenario)
    if scenario is not None:
        with open(os.path.join(output_dir, scenario_id), "wb") as handle:
            pickle.dump(scenario, handle)


def preprocess_scenarios(scenario_dir, output_dir, debug_size=None, num_parallel=8):
    """Preprocesses waymo motion data in scenario file format.

    Args:
        scenario_dir: Directory containing scenario files.
        output_dir: Directory in which to output preprocessed samples
        debug_size: If provided, limit to this number of output samples.
            This is the _max_ number of samples, but fewer may result.
        num_parallel: Number of processes to run in parallel.
            Recommend to set this to number of cores - 1.
    """
    assert os.path.exists(scenario_dir)
    filenames = os.listdir(scenario_dir)
    print(f"Saving files in {output_dir}")
    filepaths = [os.path.join(scenario_dir, f) for f in filenames]
    dataset = tf.data.TFRecordDataset(filepaths)
    os.makedirs(output_dir, exist_ok=True)

    pool = ProcessPoolExecutor(num_parallel)
    futures = []
    for i, data in enumerate(tqdm(dataset)):
        future = pool.submit(preprocess_scenario, data=data, output_dir=output_dir)
        # future = preprocess_scenario(data=data, output_dir=output_dir)
        futures.append(future)
        if debug_size is not None and i >= debug_size:
            break
    concurrent.futures.wait(futures)
    pool.shutdown()


if __name__ == "__main__":
    """
    The way this works is it provides a command line interface to the function
    where you just pass whatever arguments the function takes to the script.

    You can get a help message with:

    $ python scripts/interaction_utils/generate_dataset_waymo.py -h

    An example you might call with:

    $ python scripts/interaction_utils/generate_dataset_waymo.py \
    /path/to/scenarios/training/ /path/to/output/training --debug_size=1000 --num_parallel=48
    """
    fire.Fire(preprocess_scenarios)
