import pandas as pd
from numpy.random import random
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold, datasets
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances

real_label = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def unpickle(file):
    import pickle
    with open(f'./cifar-10-batches-py/{file}', 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def mean_distance(class_data):
    mds = manifold.MDS(2, max_iter=10)
    return mds.fit_transform(class_data)


def show_mean_distance(label_mean_distance, fig_name):
    fig, ax = plt.subplots()
    ax.scatter(label_mean_distance[:, 0], label_mean_distance[:, 1], cmap=plt.cm.Spectral)

    for i, txt in enumerate(real_label):
        ax.annotate(txt, (label_mean_distance[:, 0][i], label_mean_distance[:, 1][i]))

    plt.axis('tight')

    fig.savefig(fig_name)
    plt.close()


def compute_euclidean_distance(label_mean_distance):
    D = pairwise_distances(label_mean_distance)
    D.shape


def create_error_bar(label_error_frame):
    plt.bar(real_label, label_error_frame, align='center', alpha=0.5)
    plt.ylabel('error')
    plt.title('Programming language usage')

    plt.savefig('error_1')
    plt.close()


def part_3(all_labels, grp_dataset):
    mixed_label_error = []

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
        mixed_label_error.append(mixed_class_label_error)
    show_mean_distance(np.asarray(mixed_label_error), 'part_3_2d.png')

    print('done')


if __name__ == "__main__":
    all_data_array = None
    all_labels = None

    # for file in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']:
    for file in ['data_batch_1']:
        batch = unpickle(file)
        if all_data_array is None:
            all_data_array = batch[b'data']
        else:
            np.concatenate((all_data_array, batch[b'data']))

    # all_data_array = all_data_array.reshape(all_data_array.shape[0], 3, 32, 32)
    # labels = np.asarray(batch[b'labels'])
    # distinct_labels = np.unique(labels)
    # all_labelled_data_array = np.column_stack((all_data_array, labels))


    # all_data_array = np.transpose(all_data_array.reshape(all_data_array.shape[0],3, 32, 32), (1, 2, 0))

    all_data_frame = pd.DataFrame(all_data_array)
    all_data_frame['label'] = np.asarray(batch[b'labels'])
    grouped_dataset = all_data_frame.groupby('label')
    labels = list(grouped_dataset.groups.keys())
    label_len = len(labels)
    label_frame = {}
    label_mean_frame = None
    label_error = []
    label_pca = []

    part_3(labels, grouped_dataset)

    for i, label in enumerate(labels):
        g = grouped_dataset.get_group(label)
        g = g.drop(g.columns[[3072]], axis=1)

        # Part 2 For 10 * 10 matrix  and MDS
        mean_image = g.agg([np.mean])
        # # wrong label_frame[label] = mean_distance(g)
        if label_mean_frame is None:
            label_mean_frame = pd.DataFrame(mean_image)
        else:
            label_mean_frame = label_mean_frame.append(mean_image)



        # Show Images
        # reshaped_g = np.transpose(np.reshape(g.values, (g.shape[0], 3, 32, 32)), (0, 2, 3, 1))
        # fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
        # for j in range(5):
        #     for k in range(5):
        #         i = np.random.choice(range(len(reshaped_g)))
        #         axes1[j][k].set_axis_off()
        #         axes1[j][k].imshow(reshaped_g[i:i + 1][0])

        # Error Part 1
        # pca = PCA(n_components=20)
        # pca.fit(g)
        # # total_error = np.sum(pca.explained_variance_ratio_)
        # label_error.append(((pca.inverse_transform(pca.transform(g)) - g).pow(2)).sum().sum() / g.shape[0])








        #
        # reshaped_g = np.transpose(np.reshape(filtered, (g.shape[0], 3, 32, 32)), (0, 2, 3, 1))
        # fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
        # for j in range(5):
        #     for k in range(5):
        #         i = np.random.choice(range(len(reshaped_g)))
        #         axes1[j][k].set_axis_off()
        #         axes1[j][k].imshow(reshaped_g[i:i + 1][0])

# plt.imshow(np.transpose(mean_image.iloc[0:1].values.reshape(3, 32, 32), (1, 2, 0)))





# For Part 2
# label_mean_mds = mean_distance(label_mean_frame)
# show_mean_distance(label_mean_mds, 'part_2_2d')
# compute_euclidean_distance(label_mean_frame)

# For Error Part 1
# create_error_bar(label_error)

print('done')
