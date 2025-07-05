import random
import numpy as np


def partition_idx_labelnoniid(y, n_parties, label_num, num_classes):
    if isinstance(y, list):
        y = np.array(y)
    K = num_classes
    if label_num == K:
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, n_parties)
            for j in range(n_parties):
                net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
    else:
        loop_cnt = 0
        while loop_cnt < 1000:
            times = [0 for _ in range(num_classes)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while j < label_num:
                    ind = random.randint(0, K - 1)
                    if ind not in current:
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            if len(np.where(np.array(times) == 0)[0]) == 0:
                break
            else:
                loop_cnt += 1

        # tackle down the issue that there is zero elements in array `times`
        zero_indices = np.where(np.array(times) == 0)[0]
        for zero_time_label in zero_indices:
            client_indices = np.array([idx for idx in range(n_parties)])
            np.random.shuffle(client_indices)
            for i in client_indices:
                selected_indices_time_over_one = np.where(np.array([times[label_idx] for label_idx in contain[i]]) > 1)[
                    0]
                if len(selected_indices_time_over_one) > 0:
                    j = selected_indices_time_over_one[0]
                    times[contain[i][j]] -= 1
                    contain[i].pop(j)
                    contain[i].append(zero_time_label)
                    times[zero_time_label] += 1
                    break

        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
    return net_dataidx_map


def partition_idx_labeldir(y, n_parties, alpha, num_classes):
    min_size = 0
    min_require_size = 10
    K = num_classes
    N = y.shape[0]
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map