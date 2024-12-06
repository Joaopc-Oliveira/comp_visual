import cv2
import numpy as np
import os


class ARApp:
    def __init__(self):
        # Inicializa os modelos
        self.init_models()
        # Inicializa a captura de vídeo
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro ao acessar a câmera.")
            exit()

        # Configura a resolução da câmera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Carrega máscaras e objetos sintéticos
        self.masks = [
            cv2.imread("maskscream.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("mask_myers.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("grinch.png", cv2.IMREAD_UNCHANGED)
        ]
        self.synthetic_objects = [
            cv2.imread("grinch.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("mask_myers.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("maskscream.png", cv2.IMREAD_UNCHANGED)
        ]
        for i, mask in enumerate(self.masks + self.synthetic_objects):
            if mask is None:
                print(f"Erro: Arquivo {i + 1} não foi carregado corretamente.")
                exit()

        self.previous_face_box = None
        self.previous_object_box = None
        self.face_stationary_frames = 0
        self.object_movement_frames = 0
        self.stationary_threshold = 10
        self.movement_threshold = 10

    def init_models(self):
        # Modelo de detecção de faces SSD
        self.face_prototxt = "deploy.prototxt.txt"
        self.face_model = "res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists(self.face_prototxt) or not os.path.exists(self.face_model):
            print("Modelos de detecção de faces não encontrados.")
            exit()
        self.face_net = cv2.dnn.readNetFromCaffe(self.face_prototxt, self.face_model)

        # YOLO para detecção de objetos
        self.yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.yolo_net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]

        # Definir objetos a serem substituídos
        self.target_objects = ["bottle", "cup", "cell phone"]

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if endX - startX > 0 and endY - startY > 0:
                    faces.append((startX, startY, endX, endY))
        return faces

    def detect_objects_yolo(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo_net.setInput(blob)
        outs = self.yolo_net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]]
                detections.append({
                    'class_name': class_name,
                    'box': (x, y, x + w, y + h),
                    'confidence': confidences[i]
                })
        return detections

    def is_object_moving(self, current_box):
        if self.previous_object_box is None:
            self.previous_object_box = current_box
            self.object_movement_frames = 0
            return False

        x_diff = abs(current_box[0] - self.previous_object_box[0])
        y_diff = abs(current_box[1] - self.previous_object_box[1])

        if x_diff > 10 or y_diff > 10:
            self.object_movement_frames += 1
        else:
            self.object_movement_frames = 0

        self.previous_object_box = current_box
        return self.object_movement_frames >= self.movement_threshold

    def replace_object(self, frame, box, synthetic_obj):
        startX, startY, endX, endY = box
        obj_width = endX - startX
        obj_height = endY - startY

        # Verifica se a região é válida
        if obj_width <= 0 or obj_height <= 0:
            print("Dimensões inválidas para o objeto detectado.")
            return

        # Redimensiona o objeto sintético
        obj_resized = cv2.resize(synthetic_obj, (obj_width, obj_height))

        # Ajusta o recorte do frame para coincidir com as dimensões do objeto
        cropped_frame = frame[startY:endY, startX:endX]
        h, w, _ = cropped_frame.shape
        obj_resized = obj_resized[:h, :w]  # Garante que as dimensões coincidem

        # Aplica o alfa para sobreposição
        alpha_mask = obj_resized[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_mask

        for c in range(3):
            cropped_frame[:, :, c] = (alpha_mask * obj_resized[:, :, c] +
                                      alpha_frame * cropped_frame[:, :, c])

    def replace_face_with_mask(self, frame, box, mask_index):
        startX, startY, endX, endY = box
        face_width = endX - startX
        face_height = endY - startY

        mask = self.masks[mask_index]
        mask_resized = cv2.resize(mask, (face_width, face_height))

        alpha_mask = mask_resized[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_mask

        for c in range(3):
            frame[startY:endY, startX:endX, c] = (alpha_mask * mask_resized[:, :, c] +
                                                  alpha_frame * frame[startY:endY, startX:endX, c])

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detecta faces
            faces = self.detect_faces(frame)
            if faces:
                largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                if self.is_object_moving(largest_face):
                    mask_index = (largest_face[0] // 100) % len(self.masks)
                    self.replace_face_with_mask(frame, largest_face, mask_index)

            # Detecta objetos
            objects = self.detect_objects_yolo(frame)
            target_objs = [o for o in objects if o['class_name'] in self.target_objects]
            if target_objs:
                largest_obj = max(target_objs, key=lambda o: (o['box'][2] - o['box'][0]) * (o['box'][3] - o['box'][1]))
                if self.is_object_moving(largest_obj['box']):
                    synthetic_obj = self.synthetic_objects[0]
                    self.replace_object(frame, largest_obj['box'], synthetic_obj)

            cv2.imshow("AR App", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = ARApp()
    app.run()