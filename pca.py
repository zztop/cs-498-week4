import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.decomposition import PCA

real_label = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def unpickle(file):
    import pickle
    with open(f'./cifar-10-batches-py/{file}', 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def mean_distance(class_data):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(2, max_iter=10, random_state=seed, dissimilarity='euclidean')
    return mds.fit(class_data).dissimilarity_matrix_


def show_mean_distance(label_mean_distance, fig_name):
    pd.DataFrame(label_mean_distance).to_html('matrix_' + fig_name + '.html')
    fig, ax = plt.subplots()
    ax.scatter(label_mean_distance[:, 0], label_mean_distance[:, 1], cmap=plt.cm.Spectral)
    for i, txt in enumerate(real_label):
        ax.annotate(txt, (label_mean_distance[:, 0][i], label_mean_distance[:, 1][i]))
    plt.axis('tight')
    fig.savefig(fig_name)
    plt.close()


def create_error_bar(label_error_frame):
    plt.bar(real_label, label_error_frame, align='center', alpha=0.5)
    plt.ylabel('error')
    plt.title('Programming language usage')

    plt.savefig('error_1.png')
    plt.close()


def part_3(all_labels, grp_dataset):
    mixed_label_errors = []

    for idx, lbl in enumerate(all_labels):
        g = grouped_dataset.get_group(lbl)
        class_data = g.drop(g.columns[[3072]], axis=1)
        mean_image = class_data.agg([np.mean])
        mixed_class_label_error = []
        for inner_lbl in all_labels:
            pca = PCA(n_components=20)
            pca.fit(grp_dataset.get_group(inner_lbl).drop(g.columns[[3072]], axis=1))
            mixed_class_label_error.append(
                ((pca.inverse_transform(pca.transform(mean_image)) - mean_image).pow(2)).sum().sum() / mean_image.shape[
                    0])
        mixed_label_errors.append(mixed_class_label_error)

    mean_mixed_label_errors = []
    for idx, lbl in enumerate(all_labels):
        mean_mixed_label_error = []
        for inner_idx, inner_lbl in enumerate(all_labels):
            mean_mixed_label_error.append((mixed_label_errors[idx][inner_idx] + mixed_label_errors[inner_idx][idx]) / 2)
        mean_mixed_label_errors.append(mean_mixed_label_error)

    show_mean_distance(np.asarray(mean_mixed_label_errors), 'part_3_2d.png')

    print('done')


if __name__ == "__main__":
    all_data_array = None
    all_labels = []

    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        batch = unpickle(file)
        if all_data_array is None:
            all_data_array = batch[b'data']
            all_labels = batch[b'labels']
        else:
            all_data_array = np.vstack((all_data_array, batch[b'data']))
            all_labels = all_labels + batch[b'labels']

    # all_data_array = all_data_array.reshape(all_data_array.shape[0], 3, 32, 32)
    # labels = np.asarray(batch[b'labels'])
    # distinct_labels = np.unique(labels)
    # all_labelled_data_array = np.column_stack((all_data_array, labels))


    # all_data_array = np.transpose(all_data_array.reshape(all_data_array.shape[0],3, 32, 32), (1, 2, 0))

    all_data_frame = pd.DataFrame(all_data_array)
    all_data_frame['label'] = np.asarray(all_labels)
    grouped_dataset = all_data_frame.groupby('label')
    labels = list(grouped_dataset.groups.keys())
    label_len = len(labels)
    label_frame = {}
    label_mean_frame = None
    label_error = []
    label_pca = []

    for i, label in enumerate(labels):
        g = grouped_dataset.get_group(label)
        g = g.drop(g.columns[[3072]], axis=1)

        # ****************Error Part 1
        pca = PCA(n_components=20)
        pca.fit(g)
        # total_error = np.sum(pca.explained_variance_ratio_)
        label_error.append(((pca.inverse_transform(pca.transform(g)) - g).pow(2)).sum().sum() / g.shape[0])

        # *************** Part 2 For 10 * 10 matrix  and MDS
        mean_image = g.agg([np.mean])
        # # wrong label_frame[label] = mean_distance(g)
        if label_mean_frame is None:
            label_mean_frame = pd.DataFrame(mean_image)
        else:
            label_mean_frame = label_mean_frame.append(mean_image)
        # *****************************************************


        # Show Images
        # reshaped_g = np.transpose(np.reshape(g.values, (g.shape[0], 3, 32, 32)), (0, 2, 3, 1))
        # fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
        # for j in range(5):
        #     for k in range(5):
        #         i = np.random.choice(range(len(reshaped_g)))
        #         axes1[j][k].set_axis_off()
        #         axes1[j][k].imshow(reshaped_g[i:i + 1][0])


        # ***************For Error Part 1
    create_error_bar(label_error)

    # ************ For Part 2
    label_mean_mds = mean_distance(label_mean_frame)
    show_mean_distance(label_mean_mds, 'part_2_2d')

    # *************** PART -3************
    part_3(labels, grouped_dataset)
# ***************End Of  PART -3************





print('done')
