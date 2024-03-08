import notifier_common
import colorama

class Notifier(notifier_common.Notifier):
    def __init__(self, config):
        super().__init__(config)
        self.config.set("name", "Console Notifier")
        self.config.set(
            "description", "Displays posture alerts in the console. Useful for debugging.")
        self.config.set("id", "console")

    def notify(self, posture):
        if posture == 0:
            print(colorama.Fore.GREEN + "Good posture")
        elif posture == 1:
            print(colorama.Fore.RED + "Bad posture")