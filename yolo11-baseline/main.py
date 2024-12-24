from ultralytics import YOLO
import os
import shutil

class YOLO11Trainer:
    def __init__(self, data_yaml, model_type='n'):
        """
        Initialize YOLO11 trainer
        model_type: 'n', 's', 'm', 'l', or 'x' for different model sizes
        """
        self.data_yaml = data_yaml
        self.model_name = f"yolo11{model_type}.pt"
        self.output_dir = "runs/train"
        
    def setup_model(self):
        """Load and setup YOLO11 model"""
        # Load pretrained model
        self.model = YOLO(self.model_name)
        print(f"Loaded {self.model_name} successfully")
        
    def train(self, epochs=100, imgsz=640, batch_size=16, device='cpu'):
        """
        Train the YOLO11 model
        """
        # Start training
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            plots=True,           # Save training plots
            save=True,            # Save trained model
            save_period=10,       # Save checkpoint every 10 epochs
            cache=True,           # Cache images for faster training
            patience=50,          # Early stopping patience
            pretrained=True,      # Use pretrained weights
            optimizer='auto',     # Use default optimizer
            verbose=True,         # Print verbose output
            exist_ok=False,       # Increment run if directory exists
            resume=False,         # Resume training from last checkpoint
        )
        return results
    
    def validate(self):
        """
        Validate the trained model
        """
        metrics = self.model.val()
        return metrics
    
    def export_model(self, format='onnx'):
        """
        Export the trained model to specified format
        Supported formats: ['onnx', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs']
        """
        path = self.model.export(format=format)
        print(f"Model exported to {path}")
        return path

def main():
    # Initialize trainer with your dataset configuration
    trainer = YOLO11Trainer(
        data_yaml='dataset.yaml',  # Your dataset.yaml path
        model_type='n'            # Use nano model ('n','s','m','l','x')
    )
    
    # Setup model
    trainer.setup_model()
    
    # Train model
    results = trainer.train(
        epochs=100,
        imgsz=640,
        batch_size=16,
        device='cpu'  # Use '0' for first GPU, 'cpu' for CPU
    )
    
    # Validate model
    metrics = trainer.validate()
    print(f"Validation metrics: {metrics}")
    
    # Export model to ONNX format
    exported_path = trainer.export_model(format='onnx')

if __name__ == "__main__":
    main()