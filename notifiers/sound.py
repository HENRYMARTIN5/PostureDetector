import notifier_common
import subprocess

class Notifier(notifier_common.Notifier):
    def __init__(self, config):
        super().__init__(config)
        self.config.set("name", "Sound Notifier")
        self.config.set(
            "description", "Plays a noise when your posture is bad.")
        self.config.set("id", "sound")

    def notify(self, posture):
        if posture == 1:
            # ffplay -nodisp -autoexit notifier_resources/alert.wav
            # run non-blocking
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "notifier_resources/alert.wav"])