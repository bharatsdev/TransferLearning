import datetime
import json
import socket
import traceback
import requests

SLACK_WEBHOOK = '  # os.environ.get('');
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class SendNotification:
    def __init__(self, slackChannelName='general', slackUserName='bharat'):
        self.slackChannelName = slackChannelName;
        self.slackUserName = slackUserName;
        self.start_time = datetime.datetime.now()
        self.host_name = socket.gethostname()
        self.dump = {
            "username": "Knock Knock",
            "channel": slackChannelName,
            "icon_emoji": ":clapper:",
        }

    def publish_message(self, message):
        """
        :param message:
        :param channel:
        :param username:
        :return:
        """
        data = {"username": self.slackChannelName, "channel": self.slackChannelName, 'text': ''.join(message)}
        print(data)

        print(requests.post(SLACK_WEBHOOK, json.dumps(data)))

    def train_start(self):
        contents = ['Your training has started 🎬',
                    'Machine name: %s' % self.host_name,
                    # 'Main call: %s' % selffunc_name,
                    'Starting date: %s' % self.start_time.strftime(DATE_FORMAT)]
        # contents.append(' '.join(user_mentions))
        self.dump['text'] = '\n'.join(contents)
        self.dump['icon_emoji'] = ':clapper:'
        requests.post(SLACK_WEBHOOK, json.dumps(self.dump))

    # @slack_sender(webhook_url=SLACK_WEBHOOK, channel='everythingisdata', user_mentions='bharat')
    def push_dict(self, dictionary):
        print('[INFO] : push Notification {}'.format(dictionary))
        """
        :param dictionary: Model Training Dictionary, which you want to send to slack
        :param channelName: Channel name, where user want to push training notification
        :param username:  this is slack user name
        :return:
        """
        start_time = datetime.datetime.now()
        data = {"username": self.slackUserName, "channel": self.slackChannelName}
        values = []
        for key, val in dictionary.items():
            values.append(str(key) + " : " + str(val))
        self.dump['text'] = "\n".join(values)
        requests.post(SLACK_WEBHOOK, json.dumps(self.dump))
        # return data

    def trainCrash(self, ex):
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        contents = ["Your training has crashed ☠️",
                    'Machine name: %s' % self.host_name,
                    'Main call: %s' % self.func_name,
                    'Starting date: %s' % self.start_time.strftime(DATE_FORMAT),
                    'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                    'Crashed training duration: %s\n\n' % str(elapsed_time),
                    "Here's the error:",
                    '%s\n\n' % ex,
                    "Traceback:",
                    '%s' % traceback.format_exc()]
        # contents.append(' '.join(user_mentions))
        self.dump['text'] = '\n'.join(contents)
        self.dump['icon_emoji'] = ':skull_and_crossbones:'
        requests.post(SLACK_WEBHOOK, json.dumps(self.dump))
        raise ex


if __name__ == '__main__':
    # , channelName = 'everythingisdata', username = 'bharat'
    notify = SendNotification(slackChannelName='')
    dictI = {'loss': 0.22, 'HostName': socket.gethostname()}
    notify.push_dict(dictI)
