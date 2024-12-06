import cv2
import mediapipe as mp
from tp3 import ARApp  # Importa a classe base do TP3


class MediaPipeARApp(ARApp):
    def __init__(self):
        super().__init__()
        # Inicializa MediaPipe para detecção de mãos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Reutiliza imagens carregadas no TP3
        self.hand_overlay_image = self.synthetic_objects[0]  # Usa o primeiro objeto sintético como substituto para mãos
        self.use_mediapipe_face_detection = True  # Alternar entre MediaPipe e o modelo de detecção de faces antigo

    def detect_and_replace_hands(self, frame):
        """Detecta mãos usando MediaPipe e substitui por uma imagem."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Obtém os limites da mão
                h, w, _ = frame.shape
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Substituir mão por imagem
                self.replace_object(frame, (x_min, y_min, x_max, y_max), self.hand_overlay_image)
                print(f"Mão detectada e substituída: ({x_min}, {y_min}), ({x_max}, {y_max})")

                # Opcional: desenhar landmarks para debug
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def detect_faces_mediapipe(self, frame):
        """Detecta faces usando MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_frame)
            faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x_min = int(bboxC.xmin * iw)
                    y_min = int(bboxC.ymin * ih)
                    x_max = int((bboxC.xmin + bboxC.width) * iw)
                    y_max = int((bboxC.ymin + bboxC.height) * ih)
                    faces.append((x_min, y_min, x_max, y_max))
            return faces

    def run(self):
        """Loop principal do aplicativo."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao capturar o quadro da câmera.")
                break

            # Alternar entre MediaPipe e o modelo de detecção de faces antigo
            if self.use_mediapipe_face_detection:
                faces = self.detect_faces_mediapipe(frame)
            else:
                faces = self.detect_faces(frame)

            # Substituir a maior face imóvel por máscara
            if faces:
                largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                if not self.is_object_moving(largest_face):  # Verifica se a face está imóvel
                    mask_index = (largest_face[0] // 100) % len(self.masks)
                    self.replace_face_with_mask(frame, largest_face, mask_index)
                    print(f"Face detectada e substituída: ({largest_face[0]}, {largest_face[1]}), ({largest_face[2]}, {largest_face[3]})")

            # Detectar e substituir mãos
            self.detect_and_replace_hands(frame)

            # Exibir o quadro
            cv2.imshow("MediaPipe AR App", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MediaPipeARApp()
    app.run()
