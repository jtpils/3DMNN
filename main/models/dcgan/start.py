from dcgan import *

if __name__ == "__main__":
    print("[*] Starting DCGAN.")
    train(1000, 25, False)
    generate(10, name="dcgan>>epoch")


