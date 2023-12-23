import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type], lw=1.5, color='r', label='train acc', marker='.', markevery=2,
        #          mew=1.5)
        # loss
        plt.plot(iters, self.losses[loss_type], lw=1.5, color='g', label='train loss', marker='.', markevery=2, mew=1.5)
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], lw=1.5, color='b', label='val acc', marker='.', markevery=2,
            #          mew=1.5)
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], lw=1.5, color='darkorange', label='val loss', marker='.',
                     markevery=2, mew=1.5)

        plt.grid(True)
        plt.xlim(-0.1, 50)
        plt.ylim(-0.01, 1.01)
        plt.xlabel(loss_type)
        plt.ylabel('ACC-LOSS')
        plt.legend(loc="center right")
        plt.savefig("acc_loss.pdf")
        plt.show()

class Model_Metrics:
    def __init__(self, pred, y_test, hist):
        self.pred = pred
        self.y_test = y_test
        self.hist = hist

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(self.pred, axis=1)).ravel()

    '''
        a method to print model's metrics
        accuracy, recall, precision, and F1 score 
    '''
    def print_metrics(self):
        self.accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        print(f'Accuracy: {self.accuracy * 100:0f}%')
        fpr = self.fp / (self.fp + self.tn)
        print(f'False positive rate(FPR): {fpr * 100:0f}%')
        fnr = self.fn / (self.fn + self.tp)
        print(f'False negative rate(FN): {fnr * 100:0f}%')
        self.recall = self.tp / (self.tp + self.fn)
        print(f'Recall: {(self.tp / (self.tp + self.fn)) * 100:0f}%')
        self.precision = self.tp / (self.tp + self.fp)
        print(f'Precision: {(self.tp / (self.tp + self.fp)) * 100:0f}%')
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        print(f'F1 score: {self.f1 * 100:0f}%')

    '''
        a method to plot receiver operating characteristics (ROC)
    '''
    def plot_roc(self):
        y_scores = self.pred[:, 1]

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(self.y_test[:, 1], y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        # plt.figure(figsize=(8, 8))
        plt.rcParams['figure.figsize'] = (6, 5)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve of WIDENNET (AUC = {:.2f}%)'.format(roc_auc*100),
                 marker='.', markevery=0.05, mew=1.5)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig("roc.pdf")
        plt.show()

    '''
        a method to plot bar chart
    '''
    def plot_bar_chat(self):
        # Labels
        metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
        # Experiment labels
        experiments = ['WIDENNET', 'DR-GCN', 'GCN', 'Mythril']

        #Metrics from other models
        dr_gcn = [81.47, 80.89, 72.36, 76.39]
        gcn = [77.85, 78.79, 70.02, 74.15]
        mythril = [60.54, 71.69, 39.58, 51.02]
        oyente = [61.62, 54.71, 38.16, 44.96]

        my_values = [self.accuracy*100, self.recall*100, self.precision*100, self.f1*100]

        # Bar width
        bar_width = 0.15

        # Set up positions for bars
        index = np.arange(len(metrics))

        # Plotting
        plt.figure(figsize=(12, 6))
        # plt.grid = True
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')

        bar1 = plt.bar(index, my_values, bar_width, label='WIDENNET')
        bar2 = plt.bar(index + bar_width, dr_gcn, bar_width, label='DR-GCN')
        bar3 = plt.bar(index + 2*bar_width, gcn, bar_width, label='GCN')
        bar4 = plt.bar(index + 3 * bar_width, mythril, bar_width, label='Mythril')
        bar5 = plt.bar(index + 4 * bar_width, oyente, bar_width, label='Oyente')

        # Add labels and title
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Comparison of Metrics between Experiments')
        plt.xticks(index + bar_width / 6, metrics)
        plt.legend()

        # Show the plot
        plt.show()

    '''
        a method to plot confusion matrix
    '''
    def plot_cm(self):
        cm = np.array([[self.tn, self.fp], [self.fn, self.tp]])
        # Define classes (optional, but can be useful for labeling)
        classes = ['yes-timestamp', 'no-timestamp']
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
