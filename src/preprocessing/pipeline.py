import cv2
import numpy as np
import torch
import os
import torchvision.transforms as transforms

class SmartPreprocessor:
    """
    A smart image preprocessing pipeline for traffic surveillance cameras.
    It conditionally applies enhancements based on image quality metrics to maintain
    real-time performance.
    """
    def __init__(self, target_size=(640, 640), dark_thresh=80, blur_thresh=100.0, wash_thresh=40.0, use_zero_dce=True):
        self.target_size = target_size
        self.dark_thresh = dark_thresh
        self.blur_thresh = blur_thresh
        self.wash_thresh = wash_thresh
        
        # Initialize Deep Learning enhancements
        self.use_zero_dce = use_zero_dce
        if self.use_zero_dce:
            try:
                try:
                    from .zero_dce import enhance_net_nopool
                except ImportError:
                    from zero_dce import enhance_net_nopool
                    
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.zero_dce_model = enhance_net_nopool().to(self.device)
                
                weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'Epoch99.pth')
                if os.path.exists(weights_path):
                    self.zero_dce_model.load_state_dict(torch.load(weights_path, map_location=self.device))
                    self.zero_dce_model.eval()
                    print(f"[Init] Zero-DCE Model loaded successfully on {self.device}.")
                else:
                    print(f"[Warning] Zero-DCE weights not found at {weights_path}. Falling back to CLAHE.")
                    self.use_zero_dce = False
            except Exception as e:
                print(f"[Warning] Could not initialize Zero-DCE ({e}). Falling back to CLAHE.")
                self.use_zero_dce = False

    def assess_quality(self, image):
        """Calculates brightness, blur, and contrast scores."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = np.std(gray)
        return brightness, blur, contrast

    def enhance_low_light(self, image):
        """Enhances dark images using Zero-DCE (if enabled) or CLAHE as fallback."""
        if self.use_zero_dce:
            # Prepare image for Zero-DCE (RGB, normalize to [0,1], tensor)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                enhanced_tensor, _ = self.zero_dce_model(img_tensor)
                
            # Convert back to BGR numpy array [0, 255]
            enhanced_img = enhanced_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
            return cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        else:
            # Fallback to Fast CLAHE algorithm
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def deblur_image(self, image):
        """Applies a fast unsharp mask to restore edges."""
        gaussian_3 = cv2.GaussianBlur(image, (9, 9), 10.0)
        unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
        return unsharp_image

    def enhance_contrast(self, image):
        """Simple histogram equalization for washed-out (foggy/hazy) images."""
        # For a hackathon, we'll start with standard equalizion. Dark Channel Prior can be added later if needed.
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def letterbox(self, image, color=(114, 114, 114), return_transform=False):
        """Resizes the image to target_size while preserving aspect ratio (padding)."""
        shape = image.shape[:2]  # current shape [height, width]
        new_shape = self.target_size

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        if return_transform:
            return image, {
                "input_shape": [int(shape[0]), int(shape[1])],
                "output_shape": [int(new_shape[0]), int(new_shape[1])],
                "scale": float(r),
                "pad_left": int(left),
                "pad_top": int(top),
                "pad_right": int(right),
                "pad_bottom": int(bottom),
            }

        return image

    def process(self, image):
        """The main routing function."""
        # 1. Triage
        brightness, blur, contrast = self.assess_quality(image)
        
        processed_img = image.copy()
        
        # 2. Conditional Enhancement
        if brightness < self.dark_thresh:
            processed_img = self.enhance_low_light(processed_img)
            
        if blur < self.blur_thresh:
            processed_img = self.deblur_image(processed_img)
            
        if contrast < self.wash_thresh: # likely hazy/foggy
            processed_img = self.enhance_contrast(processed_img)

        # 3. Normalization
        final_img, letterbox_transform = self.letterbox(processed_img, return_transform=True)

        # Return standard BGR image for OpenCV and YOLO
        return final_img, {
            "brightness": brightness,
            "blur": blur,
            "contrast": contrast,
            "letterbox": letterbox_transform,
        }

# Quick test stub if run directly
if __name__ == "__main__":
    # Create a dummy image to test the pipeline runs without errors
    dummy_img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    processor = SmartPreprocessor()
    out_img, metrics = processor.process(dummy_img)
    print(f"Pipeline test successful. Input shape: {dummy_img.shape}, Output shape: {out_img.shape}")
    print(f"Metrics recorded: {metrics}")
