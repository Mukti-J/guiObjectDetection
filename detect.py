import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import traceback
import time
import shutil

def apply_enhancement(image, enhancement_type):
    """
    Aplikasikan berbagai jenis enhancement pada gambar
    
    Args:
        image: Input image (BGR format)
        enhancement_type: Jenis enhancement ('hist_eq', 'clahe', 'brightness', 'sharpen', 'denoise', None)
    
    Returns:
        Enhanced image
    """
    if enhancement_type is None:
        return image
    
    if enhancement_type == 'hist_eq':
        # Histogram Equalization
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    elif enhancement_type == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    elif enhancement_type == 'brightness':
        # Brightness & Contrast adjustment
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 20    # Brightness control (0-100)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    elif enhancement_type == 'sharpen':
        # Sharpening using kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    elif enhancement_type == 'denoise':
        # Denoising
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    return image

def load_model(weights_path, device='cpu'):
    """
    Load YOLOv5/YOLOv11 model dengan dukungan GPU/CPU
    """
    print(f"Loading model dari: {weights_path}")
    print(f"Device request dari GUI: {device}")

    # Normalize device strings: 'cuda' -> 'cuda:0', digit -> 'cuda:N'
    if isinstance(device, str):
        if device.lower() == 'cuda':
            device = 'cuda:0'
        elif device.isdigit():
            device = f'cuda:{device}'

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print("[debug] CUDA tersedia:", cuda_available)
    print(f"[debug] normalized device (requested): {device}")

    # If requested GPU but CUDA not available -> fallback to CPU
    if isinstance(device, str) and device.startswith('cuda') and not cuda_available:
        print("⚠ CUDA tidak tersedia, fallback ke CPU")
        device = 'cpu'

    # Deteksi YOLOv11 atau YOLOv5
    model_name = Path(weights_path).name.lower()
    is_v11 = ('11' in model_name or 'v11' in model_name)

    try:
        if is_v11:
            # YOLOv11
            from ultralytics import YOLO
            print("Loading YOLOv11 model...")
            t0 = time.time()
            model = YOLO(weights_path)
            print(f"[debug] YOLOv11 model instantiation time: {time.time()-t0:.2f}s")

            # Move model to device
            try:
                if isinstance(device, str) and device.startswith('cuda'):
                    print(f"[debug] Attempting to move YOLOv11 model to {device}")
                    model.to(device)
                    print(f"✓ YOLOv11 loaded on GPU ({device})")
                else:
                    print("[debug] Moving YOLOv11 model to cpu")
                    model.to('cpu')
                    print("✓ YOLOv11 loaded on CPU")
            except Exception as e:
                print(f"[error] Failed to move YOLOv11 to GPU: {e}")
                traceback.print_exc()
                model.to('cpu')
                print("⚠ Gagal memindahkan YOLOv11 ke GPU, menggunakan CPU")

            return model

        else:
            # YOLOv5
            print("Loading YOLOv5 model via torch.hub...")
            t0 = time.time()
            # Convert Path to string for torch.hub.load (torch.hub doesn't handle Path objects well)
            weights_str = str(weights_path)
            model = None
            
            # Attempt 1: Normal load
            try:
                model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=weights_str,
                    force_reload=False,
                    verbose=False
                )
            except Exception as e:
                if 'pathlib._local' in str(e) or 'pathlib' in str(e):
                    print(f"[warn] Cache corruption detected: {e}")
                    print("[debug] Clearing torch hub cache and retrying...")
                    # Clear torch hub cache
                    import shutil
                    hub_dir = Path.home() / '.cache' / 'torch' / 'hub'
                    if hub_dir.exists():
                        try:
                            shutil.rmtree(hub_dir)
                            print("[debug] Cache cleared successfully")
                        except Exception as ce:
                            print(f"[warn] Could not clear cache: {ce}")
                else:
                    print(f"[warn] First load attempt failed: {e}")
                
                # Attempt 2: Force reload
                print("[debug] Retrying with force_reload=True...")
                try:
                    model = torch.hub.load(
                        'ultralytics/yolov5',
                        'custom',
                        path=weights_str,
                        force_reload=True,
                        verbose=False
                    )
                except Exception as e2:
                    print(f"[error] Second attempt failed: {e2}")
                    raise
            
            print(f"[debug] YOLOv5 torch.hub.load time: {time.time()-t0:.2f}s")

            # Move model to deviceI
            try:
                if isinstance(device, str) and device.startswith('cuda'):
                    print(f"[debug] Attempting to move YOLOv5 model to {device}")
                    model.to(device)
                    print(f"✓ YOLOv5 loaded on GPU ({device})")
                else:
                    print("[debug] Moving YOLOv5 model to cpu")
                    model.to('cpu')
                    print("✓ YOLOv5 loaded on CPU")
            except Exception as e:
                print(f"[error] Failed to move YOLOv5 to GPU: {e}")
                traceback.print_exc()
                model.to('cpu')
                print("⚠ Gagal memindahkan YOLOv5 ke GPU, menggunakan CPU")

            return model

    except Exception as e:
        print("Error loading model:", e)
        raise

def detect_image(model, image_path, conf_thres=0.25, enhancement=None):
    """
    Deteksi objek pada gambar (support YOLOv5 dan YOLOv11)
    
    Args:
        model: YOLOv5/YOLOv11 model
        image_path: Path ke gambar input
        conf_thres: Confidence threshold
        enhancement: Jenis enhancement
    
    Returns:
        Annotated image (numpy array)
    """
    # Baca gambar
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
    
    # Apply enhancement jika ada
    if enhancement:
        img = apply_enhancement(img, enhancement)
    
    # Cek tipe model (YOLOv5 atau YOLOv11)
    model_type = type(model).__name__
    
    if 'YOLO' in model_type and hasattr(model, 'predict'):
        # YOLOv11 (ultralytics)
        results = model.predict(img, conf=conf_thres, verbose=False)
        annotated_img = results[0].plot()
    else:
        # YOLOv5
        model.conf = conf_thres
        results = model(img)
        annotated_img = results.render()[0]
    
    return annotated_img

def detect_video(model, video_path, output_path, conf_thres=0.25, enhancement=None):
    """
    Deteksi objek pada video (support YOLOv5 dan YOLOv11)
    
    Args:
        model: YOLOv5/YOLOv11 model
        video_path: Path ke video input
        output_path: Path untuk menyimpan video output
        conf_thres: Confidence threshold
        enhancement: Jenis enhancement
    
    Returns:
        Path ke video output
    """
    # Buka video input
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka video: {video_path}")
    
    # Dapatkan properties video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer    
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Cek tipe model
    model_type = type(model).__name__
    is_v11 = 'YOLO' in model_type and hasattr(model, 'predict')
    
    if not is_v11:
        # Set confidence threshold untuk YOLOv5
        model.conf = conf_thres
    
    frame_count = 0
    print(f"Processing video: {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply enhancement jika ada
        if enhancement:
            frame = apply_enhancement(frame, enhancement)
        
        # Inference
        if is_v11:
            # YOLOv11
            results = model.predict(frame, conf=conf_thres, verbose=False)
            annotated_frame = results[0].plot()
        else:
            # YOLOv5
            results = model(frame)
            annotated_frame = results.render()[0]
        
        # Tulis frame ke output
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video processing complete! Saved to: {output_path}")
    return output_path

def run(weights, source, conf_thres=0.25, enhancement=None, output_dir='runs/detect', device='cpu'):
    """
    Fungsi utama untuk menjalankan deteksi dengan dukungan GPU/CPU
    
    Args:
        weights: Path ke model weights
        source: Path ke gambar atau video input
        conf_thres: Confidence threshold (0.0 - 1.0)
        enhancement: Jenis enhancement (None, 'hist_eq', 'clahe', 'brightness', 'sharpen', 'denoise')
        output_dir: Direktori untuk menyimpan hasil
        device: Device untuk inference ('cpu', 'cuda', '0', '1', dll)
    
    Returns:
        Path ke file output
    """
    # Buat direktori output jika belum ada
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Normalize device input
    if isinstance(device, str):
        if device.lower() == 'cuda':
            device = 'cuda:0'
        elif device.isdigit():
            device = f'cuda:{device}'

    # Validate CUDA availability and (optionally) set GPU index
    if isinstance(device, str) and device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("⚠ Warning: CUDA tidak tersedia, menggunakan CPU")
            device = 'cpu'
        else:
            try:
                idx = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.set_device(idx)
                print(f"✓ Using GPU: {torch.cuda.get_device_name(idx)}")
            except Exception as e:
                print(f"⚠ Warning: Tidak dapat mengatur GPU {device} ({e}), menggunakan CPU")
                device = 'cpu'
    
    # Load model
    print(f"Loading model from: {weights}")
    model = load_model(weights, device)
    
    # Tentukan jenis file (gambar atau video)
    source_path = Path(source)
    file_ext = source_path.suffix.lower()
    
    # Generate output filename dengan timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Deteksi gambar
        print(f"Detecting objects in image: {source}")
        if enhancement:
            print(f"Applying enhancement: {enhancement}")
        
        annotated_img = detect_image(model, source_path, conf_thres, enhancement)
        
        # Simpan hasil
        output_file = output_path / f"result_{timestamp}.jpg"
        cv2.imwrite(str(output_file), annotated_img)
        print(f"Results saved to: {output_file}")
        
        return str(output_file)
    
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Deteksi video
        print(f"Detecting objects in video: {source}")
        if enhancement:
            print(f"Applying enhancement: {enhancement}")
        
        output_file = output_path / f"result_{timestamp}.mp4"
        detect_video(model, source_path, output_file, conf_thres, enhancement)
        
        return str(output_file)
    
    else:
        raise ValueError(f"Format file tidak didukung: {file_ext}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv5/YOLOv11 Object Detection dengan Enhancement dan GPU Support')
    parser.add_argument('--weights', type=str, required=True, help='Path ke model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True, help='Path ke gambar atau video input')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--enhancement', type=str, choices=['hist_eq', 'clahe', 'brightness', 'sharpen', 'denoise'], 
                        help='Jenis image enhancement')
    parser.add_argument('--output-dir', type=str, default='runs/detect', help='Direktori output')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, 0, 1, etc.)')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Jika dijalankan dari command line
    args = parse_args()
    
    try:
        output_path = run(
            weights=args.weights,
            source=args.source,
            conf_thres=args.conf_thres,
            enhancement=args.enhancement,
            output_dir=args.output_dir,
            device=args.device
        )
        print(f"\n✓ Detection completed successfully!")
        print(f"Output: {output_path}")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()