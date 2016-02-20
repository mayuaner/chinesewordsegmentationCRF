# -*- mode: Python; coding: utf-8 -*-
import numpy as np
import csv
class CRF(object):

    def __init__(self, nodedict, featuredict):
        self.nodedict = nodedict
        self.featuredict = featuredict
        num_labels = len(self.nodedict)
        num_features = len(self.featuredict)
        self.feature_parameters = np.zeros((num_labels, num_features))
        self.transition_parameters = np.zeros((num_labels, num_labels))

    def train(self, training_set, dev_set):
        """Training function"""
        self.train_sgd(training_set, dev_set, 0.001, 200)

    def train_sgd(self, training_set, dev_set, learning_rate, batch_size):
        """Minibatch SGF for training linear chain CRF"""
        num_labels = len(self.nodedict)
        num_features = len(self.featuredict)

        num_batches = len(training_set) / batch_size
        total_expected_feature_count = np.zeros((num_labels, num_features))
        total_expected_transition_count = np.zeros((num_labels, num_labels))
        print ('With all parameters = 0, the accuracy is %s' % \
                sequence_accuracy(self, dev_set))

        fudge_factor = 1e-6
        historical_feature_grad = 0
        historical_transition_grad = 0
        L1 = 0.0001
        L2 = 0.0001
        print (num_labels)
        with open("result3.csv", "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = " ", quotechar = "|", quoting = csv.QUOTE_MINIMAL)
            for i in range(3):
                for j in range(int(num_batches)):
                    batch = training_set[j*batch_size:(j+1)*batch_size]
                    total_expected_feature_count.fill(0)
                    total_expected_transition_count.fill(0)
                    total_observed_feature_count, total_observed_transition_count = self.compute_observed_count(batch)

                    for sequence in batch:
                        transition_matrices = self.compute_transition_matrices(sequence)
                        alpha_matrix = self.forward(sequence, transition_matrices)
                        beta_matrix = self.backward(sequence, transition_matrices)
                        expected_feature_count, expected_transition_count = \
                                self.compute_expected_feature_count(sequence, alpha_matrix, beta_matrix, transition_matrices)
                        total_expected_feature_count += expected_feature_count
                        total_expected_transition_count += expected_transition_count

                    feature_gradient = (total_observed_feature_count - total_expected_feature_count - self.feature_parameters * L1) / len(batch)
                    transition_gradient = (total_observed_transition_count - total_expected_transition_count - self.transition_parameters * L2) / len(batch)
                    curr_feature_grad_sum = np.sum(np.abs(feature_gradient))
                    curr_transition_grad_sum = np.sum(np.abs(transition_gradient))
                    if (i == 0 and j == 0) or (curr_feature_grad_sum < min_feature_grad_sum and curr_transition_grad_sum < min_transition_grad_sum):
                        min_feature_grad_sum = curr_feature_grad_sum
                        min_transition_grad_sum = curr_transition_grad_sum
                        backup_feature_para = self.feature_parameters
                        backup_transition_para = self.transition_parameters
                    historical_feature_grad = historical_feature_grad + feature_gradient * feature_gradient
                    historical_transition_grad = historical_transition_grad + transition_gradient * transition_gradient
                    adjusted_feature_grad = feature_gradient / (fudge_factor + np.sqrt(historical_feature_grad))
                    adjusted_transition_grad = transition_gradient / (fudge_factor + np.sqrt(historical_transition_grad))
                    self.feature_parameters += learning_rate * adjusted_feature_grad
                    self.transition_parameters += learning_rate * adjusted_transition_grad
                    acc = sequence_accuracy(self, dev_set)
                    print (acc)
                    #csvwriter.writerow([acc])
                    csvwriter.writerow([acc])
                    #csvwriter.writerow([bytes(item, 'utf8') for item in acc])
                    #csvwriter.writerow([bytes(acc,'utf8')])
            self.feature_parameters = backup_feature_para
            self.transition_parameters = backup_transition_para


    def compute_transition_matrices(self, sequence):
        """Compute transition matrices (denoted as M on the slides)

        Returns :
            a list of transition matrices
        """
        transition_matrices = []
        num_labels = len(self.nodedict)
        transition_matrix = np.zeros((num_labels, num_labels))  # This is the dummy trainsition matrix at time 0
        transition_matrices.append(transition_matrix)
        for t in range(len(sequence)):
            # compute transition matrix
            transition_matrix = np.zeros((num_labels, num_labels))
            for i in range(num_labels):                         # Label at time t-1
                for j in range(num_labels):                     # Label at time t
                    if t == 0 and i != j:
                        continue
                    transition_matrix[i, j] += self.transition_parameters[i, j]
                    for feature in sequence[t].fv:
                        transition_matrix[i, j] += self.feature_parameters[j, feature]
                    transition_matrix[i, j] = np.exp(transition_matrix[i, j])#get e^n 
            transition_matrices.append(transition_matrix)
        return transition_matrices

    def forward(self, sequence, transition_matrices):
        """Compute alpha matrix in the forward algorithm"""
        num_labels = len(self.nodedict)
        alpha_matrix = np.zeros((num_labels, len(sequence) + 1))
        for i in range(num_labels):
            alpha_matrix[i, 0] = 1                              # Initialization
        for t in range(1, len(sequence) + 1):
            transition_matrix = transition_matrices[t]
            
            for i in range(num_labels):
                for j in range(num_labels):
                    #alpha_matrix[i, t] += alpha_matrix[j, t-1] * transition_matrix[j, i]
                    alpha_matrix[i, t] += np.exp(smooth_log(alpha_matrix[j, t-1]) + smooth_log(transition_matrix[j, i]))#???
        return alpha_matrix

    def backward(self, sequence, transition_matrices):
        """Compute beta matrix in the backward algorithm"""
        num_labels = len(self.nodedict)
        beta_matrix = np.zeros((num_labels, len(sequence) + 1))
        for i in range(num_labels):
            beta_matrix[i, len(sequence)] = 1
        time = reversed(range(len(sequence)))
        #time.reverse()
        for t in time:
            transition_matrix = transition_matrices[t+1]
            for i in range(num_labels):
                for j in range(num_labels):
                    #beta_matrix[i, t] += beta_matrix[j, t+1] * transition_matrix[i, j]
                    beta_matrix[i, t] += np.exp(smooth_log(beta_matrix[j, t+1]) + smooth_log(transition_matrix[i, j]))
        return beta_matrix

    def decode(self, sequence):
        """Find the best label sequence from the feature sequence
        """
        num_labels = len(self.nodedict)
        transition_matrices = self.compute_transition_matrices(sequence)
        decoded_sequence = list(range(len(sequence)))
        backtracking = np.zeros((len(sequence) + 1, num_labels))
        viterbi_matrix = np.zeros((len(sequence) + 1, num_labels))
        for i in range(num_labels):
            viterbi_matrix[1, i] = smooth_log(transition_matrices[1][i, i])
        for t in range(2, len(sequence) + 1):
            transition_matrix = transition_matrices[t]
            for j in range(num_labels):
                viterbi_matrix[t, j] = viterbi_matrix[t-1, 0] + smooth_log(transition_matrix[0, j])
                for i in range(num_labels):
                    if viterbi_matrix[t, j] < viterbi_matrix[t-1, i] + smooth_log(transition_matrix[i, j]):
                        viterbi_matrix[t, j] = viterbi_matrix[t-1, i] + smooth_log(transition_matrix[i, j])
                        backtracking[t, j] = i
        ptr = np.argmax(viterbi_matrix[len(sequence)])
        it = reversed(range(len(sequence)))
        #it.reverse()
        for t in it:
            decoded_sequence[t] = ptr
            ptr = backtracking[t+1, ptr]
        return decoded_sequence

    def compute_observed_count(self, sequences):
        """Compute observed counts of features from the minibatch
        """
        num_labels = len(self.nodedict)
        num_features = len(self.featuredict)

        feature_count = np.zeros((num_labels, num_features))
        transition_count = np.zeros((num_labels, num_labels))
        for sequence in sequences:
            for t in range(len(sequence)):
                if t > 0:
                    transition_count[sequence[t-1].node_i, sequence[t].node_i] += 1
                feature_count[sequence[t].node_i, sequence[t].fv] += 1
        return feature_count, transition_count

    def compute_expected_feature_count(self, sequence,
            alpha_matrix, beta_matrix, transition_matrices):
        """Compute expected counts of features from the sequence

        """
        num_labels = len(self.nodedict)
        num_features = len(self.featuredict)

        feature_count = np.zeros((num_labels, num_features))
        sequence_length = len(sequence)
        Z = np.sum(alpha_matrix[:,-1])

        #gamma = alpha_matrix * beta_matrix / Z
        #gamma = np.exp(np.log(alpha_matrix) + np.log(beta_matrix) - np.log(Z))
        for t in range(sequence_length):
            for j in range(num_labels):
                #feature_count[j, sequence[t].fv] += gamma[j, t]
                feature_count[j, sequence[t].fv] += np.exp(smooth_log(alpha_matrix[j, t]) + smooth_log(beta_matrix[j, t]) - smooth_log(Z))
                #feature_count[j, sequence[t].feature_vector] += alpha_matrix[j, t] * beta_matrix[j, t] * np.exp(0 - smooth_log(Z))
        transition_count = np.zeros((num_labels, num_labels))
        for t in range(sequence_length):
            transition_matrix = transition_matrices[t]
            for i in range(num_labels):
                for j in range(num_labels):
                    transition_count[i, j] += np.exp(smooth_log(alpha_matrix[i, t-1]) + smooth_log(beta_matrix[j, t]) + smooth_log(transition_matrix[i, j]) - smooth_log(Z))
                    #transition_count[i, j] += alpha_matrix[i, t-1] * beta_matrix[j, t] / transition_matrix[i, j] / Z

        return feature_count, transition_count

def smooth_log(x):
    if x == 0:
        return -1e9
    else:
        return np.log(x)

def sequence_accuracy(sequence_tagger, test_set):
    correct = 0.0
    total = 0.0
    for sequence in test_set:
        decoded = sequence_tagger.decode(sequence)
        assert(len(decoded) == len(sequence))
        total += len(decoded)
        for i, instance in enumerate(sequence):
            if instance.node_i == decoded[i]:
                correct += 1
    return correct / total


