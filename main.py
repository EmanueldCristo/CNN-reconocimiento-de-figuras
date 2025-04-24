import argparse
import src.Capture_Triangulo as triangulo
import src.Capture_Circulo as circulo
import src.Trained_model as trainer
import src.Shape_classifier as classifier

def main():
    parser = argparse.ArgumentParser(description="Herramienta para capturar, entrenar y clasificar formas con red neuronal")
    parser.add_argument('--step', type=str, required=True, choices=[
        'capture_Triangulo', 'capture_Circulo', 'train', 'run'
    ], help="Selecciona el proceso que deseas ejecutar")

    args = parser.parse_args()

    # Mapeo de comandos
    if args.step == 'capture_Triangulo':
        print("[Proceso] Iniciando captura de imágenes: Triangulo")
        triangulo.capture_images()

    elif args.step == 'capture_Circulo':
        print("[Proceso] Iniciando captura de imágenes: Circulo")
        circulo.capture_images()

    elif args.step == 'train':
        print("[Proceso] Entrenamiento del modelo en curso...")
        trainer.train_model()

    elif args.step == 'run':
        print("[Proceso] Ejecutando clasificador de Figuras...")
        classifier.classify_image()

if __name__ == "__main__":
    # Punto de entrada principal
    main()
