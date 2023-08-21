# encoding:utf-8

import requests

token = 'xoxp-5692813899079-5709799450036-5714740791972-0931ac9d879a42adfdcf2a49403837bc'


def send_msg():
    sendurl = "https://slack.com/api/chat.postMessage"
    data = {
        "token": token,
        "channel": "@Claude",
        "text": "你好"
    }
    response=requests.post(sendurl,data=data)
    return response.text

def receive_msg():
    reveive_url="https://slack.com/api/conversations.history"
    data = {
        "token": token,
        "channel": "",
        "oldset": ""
    }
    response=requests.post(reveive_url,data=data)
    return response.text

if __name__ == '__main__':

    print(send_msg())