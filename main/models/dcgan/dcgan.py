from gan_utils import *

if __name__ == "__main__":
    print("[*] Starting DCGAN.")
    update_progress()
    model = discriminator_model()
    model.summary()

