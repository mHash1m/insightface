import os
import json
import time
import cv2
import onnxruntime as ort
import numpy as np
from numpy.linalg import norm
from pathlib import Path
import argparse
import onnxruntime as ort

from insightface.model_zoo import model_zoo
from insightface.model_zoo.arcface_onnx import *
from insightface.model_zoo.landmark import *
from insightface.model_zoo.retinaface import *
from insightface.model_zoo.attribute import *
from insightface.app.common import Face
from insightface.app.face_analysis import *


def create_model(onnx_file, model_class, providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']):
    session = model_zoo.PickableInferenceSession(onnx_file, providers = providers)
    print(f'Applied providers: {session._providers}, with options: {session._provider_options}')
    input_cfg = session.get_inputs()[0]
    input_shape = input_cfg.shape
    outputs = session.get_outputs()
    print(outputs, input_shape)
    return model_class(model_file=onnx_file, session=session)

def run_detections(img, onnx_file):
    global DET_MODEL
    if not DET_MODEL:
        retina_face = create_model(onnx_file, RetinaFace)
        retina_face.prepare(CTX_ID, det_thresh=0.5, input_size=(640, 640))
        DET_MODEL = retina_face
    boxes, kpss = DET_MODEL.detect(img, max_num=0, metric='default')
    detections = (boxes, kpss)
    return detections

def predict(models, detections, img):
    res = []
    bboxes, kpss = detections
    for i in range(bboxes.shape[0]):
        det_score = bboxes[i, 4]
        bbox = bboxes[i, 0:4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        for model in models:
            model.prepare(CTX_ID)
            model.get(img, face)
        res.append(face)
    return res

## TODO Should convert to classes structure with properties rather than global variables
def run_pipeline_on_image(orig_image):
    onnx_files = ["./models/det_10g.onnx", "./models/genderage.onnx", "./models/2d106det.onnx", "./models/1k3d68.onnx", "./models/w600k_r50.onnx"]
    classes = [RetinaFace, Attribute, Landmark, Landmark, ArcFaceONNX]
    detections = run_detections(orig_image, onnx_files[0])
    if not MODELS:
        for i in range(1, len(onnx_files)):
            MODELS.append(create_model(onnx_files[i], classes[i]))
    results = predict(MODELS, detections, orig_image)
    return results

def sim(embedding_1, embedding_2):
    cosine = np.dot(embedding_1,embedding_2)/(norm(embedding_1)*norm(embedding_2))
    return float("{:.2f}".format(cosine))

def register_faces(target):
    people = os.listdir(target)
    people_embs = dict()
    for person in people:
        person_dir = Path(os.path.join(target, person))
        imgs_list = os.listdir(person_dir)
        embedding = []
        for i in range(len(imgs_list)):
            img = cv2.imread(str(person_dir / imgs_list[i]))
            faces = app.get(img)
            embedding.append(faces[0]["embedding"])
            print(imgs_list[i], len(faces))
            rimg = app.draw_on(img, faces)
            out_dir = Path(f"./results/reg_face_outputs/{person}")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(str(out_dir / imgs_list[i]), rimg)
        people_embs[person] = np.average(embedding, axis=0).tolist()
    json.dump(people_embs, open("registered_faces.json", 'w'))

def run_for_video(target, register):
    start_time = time.time()
    cap = cv2.VideoCapture(target)
    fps, width, height = (int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = Path(target).stem
    out = cv2.VideoWriter(f"./results/{file_name}_output.mp4", fourcc, fps, (width, height))
    counter = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
                break
        faces = run_pipeline_on_image(frame)
        frame = FaceAnalysis.draw_on(None, frame, faces)
        for face in faces:
            face_emb = face["embedding"]
            highest_sim = ("mike", -1)
            for key, val in register.items():
                cos_sim = sim(face_emb, val)
                if  cos_sim > highest_sim[1]:
                    highest_sim = [key, cos_sim]
            x1, y1, x2, y2 = s = [int(px) for px in face["bbox"]]
            # print("Averaged =", highest_sim)
            if highest_sim[1] >= 0.3:
                cv2.putText(frame,f"{highest_sim[0]}[{highest_sim[1]}]",(x1,y2+12),5,0.7,(255,178,102))
        out.write(frame)
        print(f"Frame # {counter} predicted!")
        counter+=1
    print("Video Width = ", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), ", Video Height = ", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Average FPS: ", counter / (time.time() - start_time))
    cap.release()
    cv2.destroyAllWindows()

def run_for_webcam(target, register, show):
    # define a video capture object
    vid = cv2.VideoCapture(int(target))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = Path(target).stem
    out = cv2.VideoWriter(f"./results/webcam_output.mp4", fourcc, 5, (640,480))
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        faces = run_pipeline_on_image(frame)
        frame = FaceAnalysis.draw_on(None, frame, faces)
        for face in faces:
            face_emb = face["embedding"]
            highest_sim = ("mike", -1)
            for key, val in register.items():
                cos_sim = sim(face_emb, val)
                if  cos_sim > highest_sim[1]:
                    highest_sim = [key, cos_sim]
            x1, y1, x2, y2 = s = [int(px) for px in face["bbox"]]
            if highest_sim[1] >= 0.3:
                cv2.putText(frame,f"{highest_sim[0]}[{highest_sim[1]}]",(x1,y2+12),5,0.7,(255,178,102))
        # Display the resulting frame
        if show:
            cv2.imshow('frame', frame)
        out.write(frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if show and cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def run_for_image(target, register):
    file_name = Path(target).stem
    img = cv2.imread(target)
    faces = run_pipeline_on_image(img)
    rimg = FaceAnalysis.draw_on(None, img, faces)
    for face in faces:
        face_emb = face["embedding"]
        highest_sim = ("mike", -1)
        for key, val in register.items():
            cos_sim = sim(face_emb, val)
            if  cos_sim > highest_sim[1]:
                highest_sim = [key, cos_sim]
        x1, y1, x2, y2 = s = [int(px) for px in face["bbox"]]
        print("Averaged =", highest_sim)
        if highest_sim[1] >= 0.3:
            cv2.putText(rimg,f"{highest_sim[0]}[{highest_sim[1]}]",(x1,y2+12),5,0.7,(255,178,102))
    cv2.imwrite(f"./results/output_{file_name}.jpg", rimg)

parser=argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, required=True, help="'register', 'inference image' or 'inference video'")
parser.add_argument("-tg", "--target", type=str, required=True, help="path to vid, image or dir")
parser.add_argument("-s", "--show", default=False, action='store_true', help="show cam feed inference")
args=parser.parse_args()
task = args.task
target = args.target
show = args.show

CTX_ID = 0
MODELS = []
DET_MODEL = None

try:
    os.mkdir("./results")
except OSError as error:
    print(str(error)[10:])

if task == "register":
    register_faces(target)
elif task == "inference image":
    register = json.load(open("./registered_faces.json"))
    run_for_image(target, register)
elif task == "inference video":
    register = json.load(open("./registered_faces.json"))
    run_for_video(target, register)
elif task == "inference camera":
    register = json.load(open("./registered_faces.json"))
    run_for_webcam(target, register, show)
else:
    print("UNKNOWN TASK, should be 'register', 'inference image' or 'inference video'")