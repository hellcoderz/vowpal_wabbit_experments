from firebase import firebase
import time
import console
import requests
import json

firebase = firebase.FirebaseApplication('https://chitchatbot.firebaseio.com', None)
nlpbot_demo_url = "https://api.telegram.org/bot174911153:AAF2WXgU7GOo9_Lw2qkiREdpjbWciUZmqDY/sendMessage"


def process(responses):
    if responses != None and type(responses) == type({}) and len(responses) > 0:
        print "============================"
        for key in responses.keys():
            data = responses[key]
            text = data["message"]["text"]
            chat_id = data["message"]["chat"]["id"]
            print text
            print chat_id
            out = console.process(text)
            payload = {
                "chat_id": chat_id,
                "text": json.dumps(out)
            }
            res = requests.post(nlpbot_demo_url, data=payload)
            print res.text
            firebase.delete('/messages', key)


if __name__ == "__main__":
    while True:
        firebase.get_async('/messages', None, callback=process)
        time.sleep(2)
