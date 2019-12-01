import os
import telebot
from telebot import apihelper


apihelper.proxy = {'https':'socks5h://127.0.0.1:9050'}


class TgBot:
    def __init__(self):
        self.tgbotkey = '373671235:AAFpQ3V16BGTpTWU4bnnKU6k9v5m15DFy04'
        self.tbot = telebot.TeleBot(self.tgbotkey)
        self.channel = '@EXMO_Auto'

    def send(self, msg):
        self.tbot.send_message(self.channel, msg)


def notify_sound():
    file = "~/Resources/martian-gun.mp3"
    os.system("mpg123 " + file)


tgbot = TgBot()

def notify(msg, console=True, tg=True, sound=True):
    if console:
        print(msg)
    if tg:
        tgbot.send(msg)
    if sound:
        notify_sound()
