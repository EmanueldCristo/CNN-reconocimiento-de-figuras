import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

def load_trained_model(model_path="Trained_model/model.h5"):
    return load_model(model_path)

def preprocess_frame(frame, target_size=(100, 100)):
    # Redimensionar la imagen y normalizarla
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized.astype("float32") / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def classify_image():
    model = load_trained_model()
    shape_categories = ['Triangulo', 'Circulo']
    
    capture_device = cv2.VideoCapture(0)
    
    while capture_device.isOpened():
        ret, current_frame = capture_device.read()
        if not ret:
            break

        # Preprocesar la imagen antes de hacer la predicción
        input_image = preprocess_frame(current_frame)
        
        # Realizar la predicción
        prediction_probabilities = model.predict(input_image)[0]
        predicted_label = shape_categories[np.argmax(prediction_probabilities)]
        prediction_confidence = np.max(prediction_probabilities)
        
        # Mostrar el resultado en la pantalla
        display_text = f"{predicted_label} ({prediction_confidence:.2f})"
        cv2.putText(current_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar la imagen con la predicción
        cv2.imshow("Clasificador de Formas", current_frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture_device.release()
    cv2.destroyAllWindows()

