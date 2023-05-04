from torchvision import models

def download_model():
    # Load the pretrained VGG16 model
    print("downloading the model...")
    models.vgg16(pretrained=True)
    print("Done...")

if __name__ == "__main__":
    download_model()