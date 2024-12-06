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
            cv2.imread("maskscream.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("mask_myers.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("grinch.png", cv2.IMREAD_UNCHANGED)
        ]

    def init_models(self):
        # Modelo de detecção de faces SSD
        self.face_prototxt = "deploy.prototxt.txt"
        self.face_model = "res10_300x300_ssd_iter_140000.caffemodel"
        self.face_net = cv2.dnn.readNetFromCaffe(self.face_prototxt, self.face_model)

        # Modelo de pose do Mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

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
        print(f"Face detectada e substituída pela máscara {mask_index + 1}")

    def detect_gestures(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        # Implementar lógica de detecção de gestos aqui

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            if faces:
                largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                mask_index = largest_face[0] % len(self.masks)  # Critério simples para selecionar a máscara
                self.replace_face_with_mask(frame, largest_face, mask_index)

            cv2.imshow("AR App", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ARApp()
    app.run()
