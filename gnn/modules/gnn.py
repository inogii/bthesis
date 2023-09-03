import dgl
import torch
import pickle
import random
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_percentage_error, f1_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
import nni

class GNN:
    def __init__(self, task, metric, debug=False):
        self.task = task
        self.metric = metric #options: RMSE, AC
        self.debug = debug
        if task == 'link_prediction':
            if metric != 'AC':
                raise Exception('The support metric of link_prediction is `AC`')
        elif task == 'node_classification':
            if metric not in ['RMSE', 'AC']:
                raise Exception('The support metrics of node_classification are `RMSE` and `AC`')
        else:
            raise Exception('The task should be `node_classification` or `link_prediction`')
        self.graph = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.train_graph = None
        self.test_graph = None
        self.val_graph = None
        self.train_pos_graph = None
        self.train_neg_graph = None
        self.test_pos_graph = None
        self.test_neg_graph = None
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.weights = None
        self.original_weights = None
        if torch.cuda.is_available():
            #patch_sklearn()
            pass
    
    def load_original_dataset(self, data_path, force_reload=False):
        print(data_path)
        data_path = 'dataset/node_feature_export.csv'

        self.graph = dgl.data.CSVDataset('dataset', force_reload=force_reload)[0]
        f = open("dataset/original_dataset.pkl", "wb")
        pickle.dump(self.graph, f)
        f.close()
        if self.debug:
            print(self.graph)

    def load_dataset(self, name, force_reload=False):
        f = open(name, 'rb')
        self.graph = pickle.load(f)
        if self.debug:
            print(self.graph)

    @staticmethod
    def create_dataset_baseline(graph, features):
        u, v = graph.edges()
        x = []
        for i in range(len(u)):
            x.append(features[u[i].item()].tolist() + features[v[i].item()].tolist())
        return x

    def split_dataset(self, percentage, link_classification=False, classic=False):
        if self.task == 'node_classification':
            node_ids = self.graph.nodes().tolist()
            #test_size = int(len(node_ids) * percentage)
            if classic:
                last_as = len(node_ids)
            else: 
                node_types = self.graph.ndata['ntype']
                counts = torch.bincount(node_types).tolist()
                last_as = counts[1]
                last_link = counts[2]
            
            if link_classification:
                number_links = last_link
                hidden_links = int(number_links*percentage)
                print(hidden_links)
                last_hidden_link = last_as + hidden_links
                print(last_hidden_link)
                #remove last percentage of the link nodes of the train (e.g last 10%)
                self.train_graph = dgl.remove_nodes(self.graph, node_ids[last_as:last_hidden_link])
                #remove all the other links nodes from the graph (e.g the trained 90%)
                remove_test =  node_ids[:last_as] + node_ids[last_hidden_link:]
                self.test_graph = dgl.remove_nodes(self.graph,remove_test) 
            else:
                last_hidden_as = int(last_as*percentage)
                print()
                #remove last percentage of the link nodes of the train graph (e.g last 10%)
                self.train_graph = dgl.remove_nodes(self.graph, node_ids[:last_hidden_as])
                #remove the trained chunk (e.g first 90%)
                self.test_graph = dgl.remove_nodes(self.graph, node_ids[last_hidden_as:])
            
            print('train graph')
            print(self.train_graph)
            print('test graph')
            print(self.test_graph)

            X_res = []
            if len(self.get_label_shape()) > 1:
                if link_classification:
                    rus = RandomUnderSampler(random_state=42)
                    nodes_in_train = node_ids[:last_as] + node_ids[last_hidden_link:]
                    X_res, _ = rus.fit_resample(np.reshape(nodes_in_train, (-1, 1)), np.argmax(self.train_graph.ndata['label'].tolist(), axis=1))
                else:
                    rus = RandomUnderSampler(random_state=42)
                    nodes_in_train = node_ids[last_hidden_as:]
                    X_res, _ = rus.fit_resample(np.reshape(nodes_in_train, (-1, 1)), np.argmax(self.train_graph.ndata['label'].tolist(), axis=1))

            self.train_mask = []
            self.test_mask = []
            for val in self.graph.nodes().tolist():
                if type(self.graph.ndata['label'][val].tolist()) == list and np.argmax(self.graph.ndata['label'][val].tolist()) == 0:
                    self.train_mask.append(False)
                    self.test_mask.append(False)
                elif link_classification and val in node_ids[last_as:last_hidden_link]:
                    self.train_mask.append(False)
                    self.test_mask.append(True)
                elif not link_classification and val in node_ids[:last_hidden_as]:
                    self.train_mask.append(False)
                    self.test_mask.append(True)
                elif link_classification and len(self.get_label_shape()) == 1 and val in node_ids[last_hidden_link:last_link]:
                    self.train_mask.append(True)
                    self.test_mask.append(False)
                elif not link_classification and len(self.get_label_shape()) == 1 and val in node_ids[last_hidden_as:last_as]:
                    self.train_mask.append(True)
                    self.test_mask.append(False)
                elif len(self.get_label_shape()) > 1 and val in X_res.flatten():
                    self.train_mask.append(True)
                    self.test_mask.append(False)
                else:
                    self.train_mask.append(False)
                    self.test_mask.append(False)

            self.train_mask = torch.tensor(self.train_mask)
            self.test_mask = torch.tensor(self.test_mask)

            self.train_graph = self.graph

            if len(self.get_label_shape()) > 1:
                self.train_graph.ndata['label'] = np.delete(self.train_graph.ndata['label'], 0, 1)
            
            if torch.cuda.is_available():
                self.graph = self.graph.to('cuda')
                self.train_mask = self.train_mask.to('cuda')
                self.test_mask = self.test_mask.to('cuda')
                self.train_graph = self.train_graph.to('cuda')
                self.test_graph = self.test_graph.to('cuda')
    
        elif self.task == 'link_prediction':
            u, v = self.graph.edges()

            edge_ids = np.arange(self.graph.number_of_edges())
            edge_ids = np.random.permutation(edge_ids)
            test_size = int(len(edge_ids) * percentage)
            test_pos_u, test_pos_v = u[edge_ids[:test_size]], v[edge_ids[:test_size]]
            train_pos_u, train_pos_v = u[edge_ids[test_size:]], v[edge_ids[test_size:]]

            number_of_nodes = self.graph.number_of_nodes()
            adjacency = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(number_of_nodes, number_of_nodes))
            adjacency_neg = 1 - adjacency.todense() - np.eye(number_of_nodes)
            neg_u, neg_v = np.where(adjacency_neg != 0)

            neg_edge_ids = np.random.choice(len(neg_u), self.graph.number_of_edges())
            test_neg_u, test_neg_v = neg_u[neg_edge_ids[:test_size]], neg_v[neg_edge_ids[:test_size]]
            train_neg_u, train_neg_v = neg_u[neg_edge_ids[test_size:]], neg_v[neg_edge_ids[test_size:]]

            self.train_graph = dgl.remove_edges(self.graph, edge_ids[:test_size])

            self.train_pos_graph = dgl.graph((train_pos_u, train_pos_v), num_nodes=number_of_nodes)
            self.train_neg_graph = dgl.graph((train_neg_u, train_neg_v), num_nodes=number_of_nodes)

            self.test_pos_graph = dgl.graph((test_pos_u, test_pos_v), num_nodes=number_of_nodes)
            self.test_neg_graph = dgl.graph((test_neg_u, test_neg_v), num_nodes=number_of_nodes)

            train_pos = self.create_dataset_baseline(self.train_pos_graph, self.train_graph.ndata['feat'])
            train_neg = self.create_dataset_baseline(self.train_neg_graph, self.train_graph.ndata['feat'])
            train_pos_neg = pd.DataFrame(list(zip(train_pos + train_neg, torch.ones(len(train_pos)).tolist() + torch.zeros(len(train_neg)).tolist())), columns=['feat', 'label'])
            self.train_set = train_pos_neg.sample(frac=1).reset_index(drop=True)

            test_pos = self.create_dataset_baseline(self.test_pos_graph, self.train_graph.ndata['feat'])
            test_neg = self.create_dataset_baseline(self.test_neg_graph, self.train_graph.ndata['feat'])
            test_pos_neg = pd.DataFrame(list(zip(test_pos + test_neg, torch.ones(len(test_pos)).tolist() + torch.zeros(len(test_neg)).tolist())), columns=['feat', 'label'])
            self.test_set = test_pos_neg.sample(frac=1).reset_index(drop=True)


            if torch.cuda.is_available():
                self.train_graph = self.train_graph.to('cuda')

                self.train_pos_graph = self.train_pos_graph.to('cuda')
                self.train_neg_graph = self.train_neg_graph.to('cuda')

                self.test_pos_graph = self.test_pos_graph.to('cuda')
                self.test_neg_graph = self.test_neg_graph.to('cuda')

    def split_dataset_validation(self, percentage, link_classification=False, classic=False, use_undersampler=False):
        node_ids = self.graph.nodes().tolist()
        if classic:
            last_as = len(node_ids)
            last_link = 0
        else: 
            node_types = self.graph.ndata['ntype']
            counts = torch.bincount(node_types).tolist()
            last_as = counts[1]
            last_link = counts[2]
        
        if link_classification:
            number_links = last_link
            last_link = last_link + last_as
            split_size = int(number_links*percentage)
            train_list = node_ids[last_as:last_link]
        else:
            split_size = int(last_as*percentage)
            train_list = node_ids[:last_as]
    
        # Split the dataset based on the new 'labeled' attribute
        train_list = [val for val in node_ids if type(self.graph.ndata['label'][val].tolist()) == list and max(self.graph.ndata['label'][val].tolist()) != 0]
        val_list = []
        test_list = []
        while len(val_list) < split_size:
            random_element = random.choice(train_list)
            train_list.remove(random_element)
            val_list.append(random_element)

        while len(test_list) < split_size:
            random_element = random.choice(train_list)
            train_list.remove(random_element)
            test_list.append(random_element)

        self.train_graph = dgl.remove_nodes(self.graph, val_list+test_list)
        self.val_graph = dgl.remove_nodes(self.graph, test_list+train_list)
        self.test_graph = dgl.remove_nodes(self.graph, val_list+train_list)
        
        self.train_mask = []
        self.test_mask = []
        self.val_mask = []
        count_unlabeled = 0
        count_weird = 0

        # Obtaining the weights for the loss function
        node_labels = self.graph.ndata['label']
        one_hot_labels = node_labels
        label_counts = torch.sum(one_hot_labels, dim=0)
        total_nodes = torch.sum(label_counts)
        max_label = torch.max(label_counts)
        min_label = torch.min(label_counts)
        label_weights = 1 - (label_counts / total_nodes)
        self.original_weights = torch.squeeze(label_weights)
        label_weights = torch.pow(label_weights, 1)
        label_weights_tensor = torch.squeeze(label_weights)
        self.weights = label_weights_tensor
        print(label_counts)
        print(self.weights)
        X_res = [val for val in node_ids if type(self.graph.ndata['label'][val].tolist()) == list and max(self.graph.ndata['label'][val].tolist()) != 0]
        labels = [self.graph.ndata['label'][val].tolist() for val in train_list if type(self.graph.ndata['label'][val].tolist()) == list and max(self.graph.ndata['label'][val].tolist()) != 0]
        if use_undersampler:
            if len(self.get_label_shape()) > 1:
                rus = RandomUnderSampler(random_state=42)
                X_res, _ = rus.fit_resample(np.reshape(train_list, (-1, 1)), np.argmax(labels, axis=1))
            print(len(X_res))

        for val in node_ids:
            if type(self.graph.ndata['label'][val].tolist()) == list and max(self.graph.ndata['label'][val].tolist()) == 0:
                count_unlabeled += 1
                self.train_mask.append(False)
                self.val_mask.append(False)
                self.test_mask.append(False)
            elif val in train_list and val in X_res:
                self.train_mask.append(True)
                self.val_mask.append(False)
                self.test_mask.append(False)
            elif val in val_list:
                self.train_mask.append(False)
                self.val_mask.append(True)
                self.test_mask.append(False)
            elif val in test_list:
                self.train_mask.append(False)
                self.val_mask.append(False)
                self.test_mask.append(True)
            else:
                count_weird += 1
                self.train_mask.append(False)
                self.val_mask.append(False)
                self.test_mask.append(False)

        self.train_mask = torch.tensor(self.train_mask)
        self.test_mask = torch.tensor(self.test_mask)
        self.val_mask = torch.tensor(self.val_mask)

        print('Train Mask')
        print((self.train_mask.sum()))
        print('Val Mask')
        print((self.val_mask.sum()))
        print('Test Mask')
        print((self.test_mask.sum()))
        
        self.train_graph = self.graph

        if torch.cuda.is_available():
            self.graph = self.graph.to('cuda')
            self.train_mask = self.train_mask.to('cuda')
            self.test_mask = self.test_mask.to('cuda')
            self.train_graph = self.train_graph.to('cuda')
            self.test_graph = self.test_graph.to('cuda')
            self.weights = self.weights.to('cuda')
            self.val_mask = self.val_mask.to('cuda')
            self.val_graph = self.val_graph.to('cuda')
            self.original_weights = self.original_weights.to('cuda')

        

    def compute_loss(self, scores=None, labels=None, pos_score=None, neg_score=None):
        if self.task == 'link_prediction':
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        if torch.cuda.is_available():
            scores = scores.to('cuda')
            labels = labels.to('cuda')
        if self.task == 'link_prediction':
            return F.binary_cross_entropy_with_logits(scores, labels)
        else:
            return F.cross_entropy(scores, labels, weight=self.weights)

    def compute_metric(self, scores=None, labels=None, pos_score=None, neg_score=None):
        if self.task == 'link_prediction' and pos_score is not None and neg_score is not None:
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        if torch.cuda.is_available() and str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
            scores = scores.to('cpu')
            labels = labels.to('cpu')
        if self.metric == 'AC':
            if self.task == 'link_prediction':
                if str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
                    return (roc_auc_score(labels.numpy(), scores.numpy()), precision_score(labels.numpy(), np.rint(scores.numpy())), recall_score(labels.numpy(), np.rint(scores.numpy()))), (labels.numpy(), np.rint(scores.numpy()))
                else:
                    return (roc_auc_score(labels, scores), precision_score(labels, np.rint(scores)), recall_score(labels, np.rint(scores))), (labels, np.rint(scores))
            else:
                if str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
                    precision, recall, f, true_sum = f1_score(np.argmax(labels.numpy(), axis=1), np.argmax(scores.numpy(), axis=1), average='macro')
                    acc, y_true, y_pred = accuracy_score(np.argmax(labels.numpy(), axis=1), np.argmax(scores.numpy(), axis=1))
                    return (acc, f, precision, recall, y_true, y_pred), (np.argmax(labels.numpy(), axis=1), np.argmax(scores.numpy(), axis=1))
                else:
                    precision, recall, f, true_sum = f1_score(np.argmax(labels, axis=1), np.argmax(scores, axis=1), average='macro')
                    acc, y_true, y_pred = accuracy_score(np.argmax(labels, axis=1), np.argmax(scores, axis=1))
                    return (acc, f, precision, recall, y_true, y_pred), (np.argmax(labels, axis=1), np.argmax(scores, axis=1))
        elif self.metric == 'RMSE':
            if str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
                return (mean_squared_error(labels.numpy(), scores.numpy()), mean_absolute_percentage_error(labels.numpy(), scores.numpy())), (labels.numpy(), scores.numpy())
            else:
                return (mean_squared_error(labels, scores), mean_absolute_percentage_error(labels, scores)), (labels, scores)

    def train(self, model, predictor, optimizer, epochs=100):
        if torch.cuda.is_available():
            model = model.to('cuda')
            predictor = predictor.to('cuda')
        old_loss = float('-inf')
        non_decreasing_counter = 0

        for e in range(1, epochs+1):
            # forward
            h = model(self.train_graph, self.train_graph.ndata['feat'])
            if self.task == 'node_classification':
                score = predictor(self.train_graph, h)
                loss = self.compute_loss(scores=score[self.train_mask], labels=self.train_graph.ndata['label'][self.train_mask])
            elif self.task == 'link_prediction':
                pos_score, _ = predictor(self.train_pos_graph, h)
                neg_score, _ = predictor(self.train_neg_graph, h)
                loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if old_loss <= loss or (old_loss/loss)-1 < 0.001:
                non_decreasing_counter += 1
            else:
                non_decreasing_counter = 0
            
            old_loss = loss
            if non_decreasing_counter >= 10:
                print(f'Early stopping at epoch {e} because of non-decreasing loss')
                break

            if e % 5 == 0 and self.debug:
                print(f'In epoch {e}, loss: {loss}')

    def train_validation(self, model, predictor, optimizer, learning_rate_decay, patience, epochs=100):
        if torch.cuda.is_available():
            model = model.to('cuda')
            predictor = predictor.to('cuda')
        old_loss = float('-inf')
        non_decreasing_counter = 0
        
        train_loss_list = []
        val_loss_list = []
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_decay, patience=patience, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
        for e in range(1, epochs+1):
            # forward
            h = model(self.train_graph, self.train_graph.ndata['feat'])
            score = predictor(self.train_graph, h)
            train_loss = self.compute_loss(scores=score[self.train_mask], labels=self.train_graph.ndata['label'][self.train_mask])
            val_loss = self.compute_loss(scores=score[self.val_mask], labels=self.train_graph.ndata['label'][self.val_mask])
            
            train_loss_list.append(train_loss.item())
            val_loss_list.append(val_loss.item())
            # backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step(val_loss)

            val_metric, res = self.score_val(model, predictor)

            if old_loss <= val_loss or (old_loss/val_loss)-1 < 0.0001:
                non_decreasing_counter += 1
            else:
                non_decreasing_counter = 0
            
            old_loss = val_loss

            if non_decreasing_counter >= 70:
                print(f'Early stopping at epoch {e} because of non-decreasing validation loss')
                break
            nni.report_intermediate_result(val_metric[2])
            if self.debug:
                print(f'In epoch {e}, train_loss: {train_loss}, val_loss: {val_loss}, acc: {val_metric[0]}, precision: {val_metric[2]}')

        return train_loss_list, val_loss_list

    def train_with_embeddings(self, model, optimizer, epochs=100):
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        for e in range(1, epochs+1):
            if self.task == 'node_classification':
                score = model(self.train_graph, self.train_graph.ndata['feat'])
                loss = self.compute_loss(scores=score[self.train_mask], labels=self.train_graph.ndata['label'][self.train_mask])
            elif self.task == 'link_prediction':
                pos_score = model(self.train_pos_graph, self.train_graph.ndata['feat'])
                neg_score = model(self.train_neg_graph, self.train_graph.ndata['feat'])
                loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if e % 5 == 0 and self.debug:
                print(f'In epoch {e}, loss: {loss}')

    def train_with_baseline(self, model):
        return 0

    def score(self, model, predictor):
        if torch.cuda.is_available():
            model = model.to('cuda')
            predictor = predictor.to('cuda')
        with torch.no_grad():
            h = model(self.train_graph, self.train_graph.ndata['feat'])
            if self.task == 'node_classification':
                score = predictor(self.train_graph, h)
                if self.metric == 'RMSE':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                elif self.metric == 'AC':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                return computed_metric, res
            elif self.task == 'link_prediction':
                pos_score, (pos_src, pos_dst) = predictor(self.test_pos_graph, h)
                neg_score, (neg_src, neg_dst) = predictor(self.test_neg_graph, h)
                nodes_src = torch.cat([pos_src, neg_src]).tolist()
                nodes_dst = torch.cat([pos_dst, neg_dst]).tolist()
                if self.metric == 'AC':
                    computed_metric, res = self.compute_metric(pos_score=pos_score, neg_score=neg_score)
                return computed_metric, res, (nodes_src, nodes_dst)
    
    def score_val(self, model, predictor):
        if torch.cuda.is_available():
            model = model.to('cuda')
            predictor = predictor.to('cuda')
        with torch.no_grad():
            h = model(self.train_graph, self.train_graph.ndata['feat'])
            if self.task == 'node_classification':
                score = predictor(self.train_graph, h)
                if self.metric == 'RMSE':
                    computed_metric, res = self.compute_metric(scores=score[self.val_mask], labels=self.train_graph.ndata['label'][self.val_mask])
                elif self.metric == 'AC':
                    computed_metric, res = self.compute_metric(scores=score[self.val_mask], labels=self.train_graph.ndata['label'][self.val_mask])
                return computed_metric, res
            elif self.task == 'link_prediction':
                pos_score, (pos_src, pos_dst) = predictor(self.test_pos_graph, h)
                neg_score, (neg_src, neg_dst) = predictor(self.test_neg_graph, h)
                nodes_src = torch.cat([pos_src, neg_src]).tolist()
                nodes_dst = torch.cat([pos_dst, neg_dst]).tolist()
                if self.metric == 'AC':
                    computed_metric, res = self.compute_metric(pos_score=pos_score, neg_score=neg_score)
                return computed_metric, res, (nodes_src, nodes_dst)

    def score_with_embeddings(self, model):
        if torch.cuda.is_available():
            model = model.to('cuda')
        with torch.no_grad():
            if self.task == 'node_classification':
                score = model(self.train_graph, self.train_graph.ndata['feat'])
                if self.metric == 'RMSE':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                elif self.metric == 'AC':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                return computed_metric, res
            elif self.task == 'link_prediction':
                pos_score = model(self.test_pos_graph, self.train_graph.ndata['feat'])
                neg_score = model(self.test_neg_graph, self.train_graph.ndata['feat'])
                if self.metric == 'AC':
                    computed_metric, res = self.compute_metric(pos_score=pos_score, neg_score=neg_score)
            return computed_metric, res

    def score_with_baseline(self, model):
        return 0

    def get_train_shape(self):
        return self.train_graph.ndata['feat'].shape

    def get_label_shape(self):
        return self.train_graph.ndata['label'].shape
