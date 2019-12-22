import json
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    def __init__(self, filePath, jsonPath=None, startAt=0, notificaiton=None):
        # Store the output path for the figure, the path to the json serialized file and the starting epchs
        super(TrainingMonitor, self).__init__()
        self.filePath = filePath
        self.jsonPath = jsonPath
        self.startAt = startAt
        print('File Path :{} , JsonPath : {}  startAt : {}'.format(self.filePath, self.jsonPath, self.startAt))
        self.slackSendNotification = notificaiton

    def on_train_begin(self, logs={}):
        self.slackSendNotification.push_dict({})
        print('[INFO] : OnTrainBegin : {}'.format(logs))
        # Initialize the history dictionary
        self.H = {}

        # if the JSON history path exist, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    for k in self.H.keys:
                        # Loopover  the entries in the history log
                        # and  trim any entries that are past the starting epochs
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        print('[INFO] : on_epoch_end Epoch : {} , logs : {}'.format(epoch, logs))
        self.slackSendNotification.push_dict(logs)

        # loop over the logs and update the losses, accuracy etc. for the entire training units
        for (key, val) in logs.items():
            l = self.H.get(key, [])
            l.append(float(val))
            self.H[key] = l

        print('[INFO] : history : {}'.format(self.H))

        # check to see if the training  history should be serialized to file
        if self.jsonPath is not None:
            with open(self.jsonPath, 'w') as f:
                f.write(json.dumps(self.H))

        if len((self.H['loss'])) > 1:
            # plot training loss and accuracy
            N = np.arange(0, len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], labels='Train_loss')
            plt.plot(N, self.H['val_loss'], labels='val_loss')
            plt.plot(N, self.H['acc'], labels='Train_acc')
            plt.plot(N, self.H['val_acc'], labels='val_acc')
            plt.title('Training Loss & Accuracy [Epoch : {}]'.format(len(self.H['loss'])))
            plt.xlabel('Loss/Accuracy')
            plt.ylabel('Epoch #')
            plt.legend()

            # Save the figure
            plt.savefig(self.filePath)
            plt.close()

    def on_train_end(self, logs=None):
        print('ddddd', logs)
