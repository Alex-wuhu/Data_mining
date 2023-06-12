
import os
import math
import numpy as np
import pandas as pd
import argparse
NOISE = -1


class DBSCAN:
    def __init__(self, input_data_frame, n_cluster, Eps, MinPts):
        self.input_data = input_data_frame.values
        self.num_of_cluster = n_cluster
        self.Eps = Eps
        self.MinPts = MinPts
        self.point_list = [Point(int(data[0]), float(data[1]), float(data[2])) for data in self.input_data]
        self.cluster_label = 0

    def make_neighbor_list(self, core):
        return [point for point in self.point_list if point != core and core.calculate_distance(point) <= self.Eps]

    def retrieve_cluster(self, neighbor_list, label):
        for neighbor in neighbor_list:
            if neighbor.label is None:
                neighbor.label = label
                other_neighbor_list = self.make_neighbor_list(neighbor)
                if self.MinPts <= len(other_neighbor_list):
                    neighbor_list.extend(other_neighbor_list)
            elif neighbor.label == NOISE:
                neighbor.label = label

    def make_cluster_list(self):
        cluster_list = [[] for _ in range(self.cluster_label)]
        for point in self.point_list:
            if point.label == NOISE:
                continue
            cluster_list[point.label].append(point.object_id)

        cluster_list.sort(key=len, reverse=True)
        return cluster_list[:self.num_of_cluster]

    def clustering(self):
        for point in self.point_list:
            if point.label is None:
                neighbors = self.make_neighbor_list(point)
                if len(neighbors) < self.MinPts:
                    point.label = NOISE
                    continue

                point.label = self.cluster_label
                self.retrieve_cluster(neighbors, self.cluster_label)
                self.cluster_label += 1


class Point:
    def __init__(self, object_id, x_pos, y_pos):
        self.label = None
        self.object_id = object_id
        self.x_pos = x_pos
        self.y_pos = y_pos

    def calculate_distance(self, point):
        return math.sqrt((self.x_pos - point.x_pos) ** 2 + (self.y_pos - point.y_pos) ** 2)


def read_file(input_filename):
    header = ['object_id', 'x_coord', 'y_coord']
    input_data_frame = pd.read_csv(input_filename, sep='\t', names=header)
    return input_data_frame


def create_directory(directory_name):
    try:
        os.makedirs(directory_name, exist_ok=True)
    except OSError:
        print("Error: Failed to create the directory.")


def main(input_filename, n, eps, min_pts):
    input_data_frame = read_file(input_filename)
    dbscan = DBSCAN(input_data_frame, n, eps, min_pts)
    dbscan.clustering()
    clusters = dbscan.make_cluster_list()

    directory_name = input_filename.replace('.txt', '') + '_result'
    create_directory(directory_name)
    for i, cluster in enumerate(clusters):
        output_file_name = input_filename.replace('.txt', '') + f"_cluster_{i}.txt"
        output_file_path = os.path.join(directory_name, output_file_name)

        with open(output_file_path, "w") as output_file:
            output_file.write('\n'.join(str(object_id) for object_id in cluster))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_filename",
                        help="input data file's name (not clustered)",
                        type=str)
    parser.add_argument("num_of_cluster",
                        help="number of clusters for the corresponding input data",
                        type=int)
    parser.add_argument("max_radius",
                        help="maximum radius of the neighborhood",
                        type=float)
    parser.add_argument("min_num_of_points",
                        help="minimum number of points in an Eps-neighborhood of a given data",
                        type=float)

    args = parser.parse_args()
    main(args.input_data_filename, args.num_of_cluster, args.max_radius, args.min_num_of_points)