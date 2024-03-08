import notifier_common
import requests
import threading

class Notifier(notifier_common.Notifier):
    def __init__(self, config):
        super().__init__(config)
        self.config.set("name", "Web Call Notifier")
        self.config.set(
            "description", "Sends a request to a web endpoint when your posture is bad.")
        self.config.set("id", "webcall")
        self.config.create("url", "https://example.com", "URL", "URL to send GET request to.")

    def request_thread(self, url):
        requests.get(url)

    def notify(self, posture):
        if posture == 1:
            threading.Thread(target=self.request_thread, args=(self.config.get_user("url"),)).start()
