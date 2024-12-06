import cv2
import numpy as np
import mediapipe as mp
import os

class ARApp:
    def __init__(self):
        # Inicializa os modelos
        self.init_models()
        # Inicializa a captura de vídeo
        self.cap = cv2.VideoCapture(0)
        # Carrega máscaras de Halloween e Natal
        self.masks = [
            cv2.imread("halloween_mask1.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("halloween_mask2.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("christmas_mask1.png", cv2.IMREAD_UNCHANGED)
        ]
        # Variáveis para controle de máscara
        self.current_mask_index = None
        self.mask_counter = 0
        self.mask_duration = 30  # Número de quadros para manter a mesma máscara

    def init_models(self):
        # Modelo de detecção de faces SSD
        self.face_prototxt = "deploy.prototxt.txt"
        self.face_model = "res10_300x300_ssd_iter_140000.caffemodel"
        self.face_net = cv2.dnn.readNetFromCaffe(self.face_prototxt, self.face_model)

        # Modelo de pose do Mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Modelo de detecção de objetos YOLOv4
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
        return faces

    def replace_face_with_mask(self, frame, box, mask_index):
        startX, startY, endX, endY = box
        face_width = endX - startX
        face_height = endY - startY

        # Seleciona uma máscara com base no índice
        mask = self.masks[mask_index]
        if mask is None:
            print(f"Máscara {mask_index + 1} não encontrada.")
            return
        mask_resized = cv2.resize(mask, (face_width, face_height))

        # Adiciona a máscara ao frame
        alpha_mask = mask_resized[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_mask

        for c in range(0, 3):
            frame[startY:endY, startX:endX, c] = (alpha_mask * mask_resized[:, :, c] +
                                                  alpha_frame * frame[startY:endY, startX:endX, c])
        cv2.putText(frame, f"Face detectada e substituída pela máscara {mask_index + 1}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def detect_gestures(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            # Um braço no ar
            if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
                cv2.putText(frame, "Um braço no ar detectado!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Dois braços no ar
            if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                cv2.putText(frame, "Dois braços no ar detectados!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Swipe left
            if right_wrist.x < right_shoulder.x:
                cv2.putText(frame, "Swipe left detectado!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Swipe right
            if left_wrist.x > left_shoulder.x:
                cv2.putText(frame, "Swipe right detectado!", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def detect_objects(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

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

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        largest_area = 0
        largest_box = None

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_box = (x, y, w, h)

        if largest_box is not None:
            x, y, w, h = largest_box
            cv2.putText(frame, "Objeto detectado e substituído!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detecta faces no frame
            faces = self.detect_faces(frame)

            # Se houver faces detectadas, escolhe uma máscara para substituir
            if faces:
                print("Face detectada!")
                largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

                # Atualiza o índice da máscara a cada 'mask_duration' quadros
                if self.mask_counter % self.mask_duration == 0:
                    self.current_mask_index = (self.current_mask_index + 1) % len(self.masks) if self.current_mask_index is not None else 0

                # Substitui a face pela máscara selecionada
                self.replace_face_with_mask(frame, largest_face, self.current_mask_index)
                self.mask_counter += 1

            # Detecta gestos no frame
            self.detect_gestures(frame)

            # Detecta objetos no frame
            self.detect_objects(frame)

            # Mostra o frame resultante
            cv2.imshow("Deteção de Faces com Máscaras e Objetos", frame)

            # Sai do loop se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera a captura de vídeo e fecha todas as janelas
        self.cap.release()
        cv2.destroyAllWindows()

# Executa o aplicativo
if __name__ == "__main__":
    app = ARApp()
    app.run()
