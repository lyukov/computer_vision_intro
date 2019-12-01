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
