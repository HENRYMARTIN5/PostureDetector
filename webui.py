import notifier_common
import os
import threading
import schedule
from fastapi import FastAPI, Request, Response, HTTPException
import posture_realtime as pr
import cv2
import datetime, time
import traceback
import json
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
templates = Jinja2Templates(directory="templates")
from betterlib.logging import Logger
import uvicorn

# init log first so we can show that we're alive
log = Logger("./webui.log", "PostureWebUI")

log.info('Loading modules...')

app = FastAPI()

config = pr.config

log.info('Loading model...')
model = pr.get_testing_model()
model.load_weights('./model/keras/model.h5')
pr.model = model  # ughhhhh this is so idiotic

# log.info out possible video inputs
avail_cams = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        avail_cams.append(i)
        cap.release()
log.info("Available cameras: " + str(avail_cams))
defcam = config.get("webui").get("camera_index", 0)
log.debug("Defaulting to camera " + str(defcam))
camera_num = defcam
cap = cv2.VideoCapture(camera_num)
vi = cap.isOpened()

msg = False
posture = False
running = True
HEADLESS = True

if vi:
    # cap.set(100, 160)
    # cap.set(200, 120)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # width = 1920
    # height = 1080
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    settings = {
        cv2.CAP_PROP_FRAME_WIDTH: 224,
        cv2.CAP_PROP_FRAME_HEIGHT: 224,
        cv2.CAP_PROP_FPS: 5
    }
    success = True
    for key, value in settings.items():
        success = cap.set(key, value) and success
    if not success:
        log.warn("Unable to set some of (or all) camera settings!")
else:
    log.warn("Unable to open camera! Please check your camera connection or select another camera from the WebUI.")
    cap.release()

log.info('Loading notifiers...')
files = os.listdir("./notifiers")
notifiers = []
for file in files:
    if file.endswith(".py"):
        # ?!? what the hell
        notifiers.append(__import__("notifiers." + file[:-3], fromlist=["notifiers"]).Notifier(
            notifier_common.NotifierConfig(source_config=(config, "notifiers." + file[:-3]))))
        log.info("Loaded notifier: " + file[:-3])
current_frame = b''

global_frame = None

# def get_camera():
#     global cap, global_frame
#     ret, frame = cap.read()
#     if not ret:
#         log.error("Unable to read frame from camera")
#         return
#     global_frame = frame

def schedule_thread():
    global running
    threads = []
    thread_count = 0
    thread_target = 10
    #schedule.every(0.1).seconds.do(get_camera)
    while running:
        #schedule.run_pending()
        try:
            # if thread_count < thread_target:
            #     threads.append(threading.Thread(target=generate))
            #     threads[-1].start()
            #     thread_count += 1
            # else:
            #     for t in threads:
            #         t.join()
            #     threads = []
            #     thread_count = 0
            generate()
        except KeyboardInterrupt:
            log.info("Shutting down...")
            running = False
            break    
        except Exception as e:
            log.error("Unable to generate frame: " +
                      str(traceback.format_exc()))

def send_notification(posture):
    for notifier in notifiers:
        try:
            notifier._notify(posture)
        except KeyboardInterrupt:
            log.info("Shutting down...")
            running = False
            break
        except Exception as e:
            log.error("Unable to send notification: " +
                      str(traceback.format_exc()))




def generate():
    global model, msg, posture, cap, current_frame, global_frame, params, model_params
    if not cap.isOpened():
        msg = "Unable to open camera! Please check your camera connection or select another camera from the WebUI."
        current_frame = b''
        return
    ret, frame = cap.read()
    canvas, position = pr.process(frame, params, model_params)
    log.info("Current back position: " + str(position))
    if not HEADLESS:
        cv2.imshow("capture", canvas)
    encoded = cv2.imencode('.png', canvas)
    if position == None:
        msg = "Camera is not in a lateral view of the body or is otherwise unable to detect the ears and hips. Please adjust the camera and try again."
    else:
        msg = False
    posture = position
    if encoded[0]:
        current_frame = encoded[1].tobytes()
    else:
        current_frame = b''
    send_notification(position)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/get_msg')
def get_msg():
    global msg
    if msg:
        return {"msg": msg, "posture": posture}
    return {"msg": "nope", "posture": posture}


@app.get('/get_posture')
def get_posture():
    global posture
    if posture:
        return posture
    return "nope"


@app.get('/get_cameras')
def get_cameras():
    global avail_cams
    return avail_cams


@app.get('/video_feed')
def video_feed():
    return Response(content=current_frame, media_type="image/png")


@app.get('/reload_cameras')
def reload_cameras():
    global avail_cams
    avail_cams = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            avail_cams.append(i)
            cap.release()
    return avail_cams


@app.get('/get_notifiers')
def get_notifiers():
    global notifiers
    return [str(n) for n in notifiers]


@app.post('/set_config')
def set_config(request: Request):
    global config, cap
    data = request.get_json()
    sections = {
        "params": ["use_gpu", "GPUdeviceNumber"],
        "webui": ["camera_index", "posture_thresholds"]
    }
    old_webui = config.get("webui")
    log.debug(str(data))
    for key, value in data.items():
        # log.debug("Setting " + key + " to " + value)
        if key in sections.get("params"):
            new = config.get("params")
            new[key] = value
            config.set("params", new)
        elif key in sections.get("webui"):
            new = config.get("webui")
            new[key] = value
            config.set("webui", new)
    cap.release()
    cap = cv2.VideoCapture(config.get("webui").get("camera_index"))
    return "ok"


@app.get('/notifier_config/getall')
def notifier_config_getall():
    # notifiers in config are defined as "notifiers.<notifier id>", so we need to strip off the start bit and only send values following this format
    keys = config.keys()
    notifier_keys = []
    for key in keys:
        if key.startswith("notifiers."):
            notifier_keys.append(key)
    notifiers = []
    for key in notifier_keys:
        notifiers.append(config.get(key))
    return notifiers

@app.post('/notifier_config/setall')
def notifier_config_setall(request: Request):
    data = request.json()
    # {'test': {'enabled': False, 'testkey': 'testval', 'testcheck': '1'}}
    for k, v in data.items():
        old = config.get("notifiers." + k)
        for k2, v2 in v.items():
            if k2 == "enabled":
                old[k2] = v2
            else:
                old[k2]["value"] = v2
        config.set("notifiers." + k, old)
    for notifier in notifiers:
        notifier.reload_config()
    return "ok"


@app.get('/get_config/{section}/{key}')
def get_config(section: str, key: str):
    section_config = config.get(section)
    if not section_config:
        raise HTTPException(status_code=404, detail="Section not found")
    if key not in section_config:
        raise HTTPException(status_code=404, detail="Key not found in section")
    return section_config[key]

@app.get('/get_config/{section}')
def get_config_section(section: str):
    section_config = config.get(section)
    if not section_config:
        raise HTTPException(status_code=404, detail="Section not found")
    return section_config

@app.on_event("startup")
async def startup_event():
    global params, model_params, running
    params = config.get("params")
    model_params = config.get("model")    
    t = threading.Thread(target=schedule_thread)
    t.start()

@app.on_event("shutdown")
async def shutdown_event():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)