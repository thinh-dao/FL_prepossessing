import numpy as np
import random
import tensorflow as tf 
from tensorflow.keras import layers

class Distribution:
    def __init__(self, dataset, num_users):
        if dataset == "MNIST":
            self.train_data, self.test_data = self.MNIST()
        elif dataset == "FashionMNIST":
            self.train_data, self.test_data = self.FashionMNIST()
        elif dataset == "CIFAR10":
            self.train_data, self.test_data = self.CIFAR10()
        elif dataset == "CIFAR100":
            self.train_data, self.test_data = self.CIFAR100()
        else:
            raise Exception("Dataset not found")

        self.num_data = len(self.train_data[0])
        self.num_classes = len(self.train_data[1][0])
        self.nitems_per_class = int(self.num_data/self.num_classes)
        self.num_users = num_users

    def iid(self):
        "balanced number of datasets and classes"
        mu_clients = 1/self.num_users
        client_distribution = np.random.normal(mu_clients, mu_clients/1000, size = self.num_users)
        client_samples = np.random.multinomial(self.num_data, client_distribution)

        dict_users = dict()
        dict_items = dict()
        for i in range(self.num_classes):
            idx = self.nitems_per_class * i
            dict_items[i] = set(range(idx, self.nitems_per_class + idx))

        mu_classes = 1/self.num_classes
        for i in range(self.num_users - 1):
            data_distribution = np.random.normal(mu_classes, mu_classes/100, size = self.num_classes)
            client_data_samples = np.random.multinomial(client_samples[i], data_distribution)
            for j in range(self.num_classes):
                if i not in dict_users: 
                    dict_users[i] = set()
                addition = set(np.random.choice(list(dict_items[j]), client_data_samples[j], replace=False))
                dict_users[i] |= addition
                dict_items[j] -= addition

        dict_users[self.num_users - 1] = set()
        for j in range(self.num_classes):
            dict_users[self.num_users - 1] |= dict_items[j]

        return dict_users

    def noniid(self):
        "Imbalanced datasets with equal number of classes"
        dict_users = dict()
        for i in range(self.num_users):
            dict_users[i] = set()

        dict_items = dict()
        for i in range(self.num_classes):
            idx = self.nitems_per_class * i
            dict_items[i] = set(range(idx, self.nitems_per_class + idx))
        
        for i in range(self.num_classes):
            label = self.labels[i]
            client_distribution = np.random.dirichlet(random.sample(range(1, 101), self.num_users))
            client_samples = np.random.multinomial(self.nitems_per_class, client_distribution)
            for j in range(self.num_users):
                dict_users[j] |= set(np.random.choice(list(dict_items[label]), client_samples[j], replace=False))

        return dict_users

    def nonIID_unequal(self):
        "Imbalanced datasets with unequal number of classes"
        dict_users = dict()
        for i in range(self.num_users):
            dict_users[i] = set()
            
        dict_items = dict()
        for i in range(self.num_classes):
            idx = self.nitems_per_class * i
            dict_items[i] = set(range(idx, self.nitems_per_class + idx))
        
        for i in range(self.num_classes):
            label = self.labels[i]
            num_users = random.randint(int(self.num_classes/2), self.num_classes)
            client_distribution = np.random.dirichlet(random.sample(range(1, 101), num_users))
            clients_selection = random.sample(range(self.num_users), num_users)
            client_samples = np.random.multinomial(self.nitems_per_class, client_distribution)
            for j in range(num_users):
                user = clients_selection[j]
                dict_users[user] |= set(np.random.choice(list(dict_items[label]), client_samples[j], replace=False))
                  
        return dict_users

    def extreme_nonIID(self):
        "Extremely non-balanced datasets with unequal number of classes"
        dict_users = dict()
        for i in range(self.num_users):
            dict_users[i] = set()

        dict_items = dict()
        for target in self.labels:
            idx = self.labels.index(target)
            dict_items[target] = set(range(self.nitems_per_class * idx, self.nitems_per_class * (idx+1)))
        
        for i in range(self.num_classes):
            label = self.labels[i]
            num_users = random.randint(1, int(self.num_classes/2))
            client_distribution = np.random.dirichlet(random.sample(range(1, 1001), num_users))
            clients_selection = random.sample(range(self.num_users), num_users)
            client_samples = np.random.multinomial(self.nitems_per_class, client_distribution)
            for j in range(num_users):
                user = clients_selection[j]
                dict_users[user] |= set(np.random.choice(list(dict_items[label]), client_samples[j], replace=False))
                  
        return dict_users

    def MNIST(self):
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = (train_images / 255.0).astype(np.float32)
        test_images = (test_images / 255.0).astype(np.float32)
        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)

        return (train_images, train_labels), (test_images, test_labels)
    
    def CIFAR10(self):
        cifar10 = tf.keras.datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        train_images = (train_images / 255.0).astype(np.float32)
        test_images = (test_images / 255.0).astype(np.float32)
        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)
        return (train_images, train_labels), (test_images, test_labels)
