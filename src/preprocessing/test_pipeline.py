import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from pipeline import SmartPreprocessor

def plot_before_after(original, processed, metrics, title="Enhancement"):
    # Convert BGR to RGB for matplotlib
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_rgb)
    axes[0].set_title(f"Original\nBright: {metrics['brightness']:.1f}, Blur: {metrics['blur']:.1f}")
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title("Processed (Letterboxed & Enhanced)")
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs('problem/output_samples', exist_ok=True)
    plt.savefig(f"problem/output_samples/{title.replace(' ', '_').replace('/', '_')}.png")
    plt.close()

def main():
    processor = SmartPreprocessor()
    
    # 1. Test a Dark Image from ExDark
    exdark_images = glob.glob("../../dataset/ExDark/**/*.jpg", recursive=True) + \
                    glob.glob("../../dataset/ExDark/**/*.png", recursive=True) + \
                    glob.glob("dataset/ExDark/**/*.jpg", recursive=True) + \
                    glob.glob("dataset/ExDark/**/*.png", recursive=True)

    
    if exdark_images:
        test_img_path = exdark_images[0]
        img = cv2.imread(test_img_path)
        if img is not None:
            processed_img, metrics = processor.process(img)
            plot_before_after(img, processed_img, metrics, title=f"ExDark Sample")
            print(f"Successfully processed ExDark image. Output saved to problem/output_samples/ExDark_Sample.png")
    else:
        print("No ExDark images found. Check the dataset path.")

    # 2. Test a Weather Image from DAWN
    dawn_images = glob.glob("../../dataset/DAWN/**/*.jpg", recursive=True) + \
                  glob.glob("../../dataset/DAWN/**/*.png", recursive=True) + \
                  glob.glob("dataset/DAWN/**/*.jpg", recursive=True) + \
                  glob.glob("dataset/DAWN/**/*.png", recursive=True)
                  
    if dawn_images:
        test_img_path = dawn_images[0]
        img = cv2.imread(test_img_path)
        if img is not None:
            processed_img, metrics = processor.process(img)
            plot_before_after(img, processed_img, metrics, title=f"DAWN Sample")
            print(f"Successfully processed DAWN image. Output saved to problem/output_samples/DAWN_Sample.png")
    else:
        print("No DAWN images found. Check the dataset path.")

if __name__ == "__main__":
    main()
