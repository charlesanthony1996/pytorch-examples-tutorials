import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt



model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)

images = model.text_to_image(
    "cute magical flying dog , fantasy art",
    "golden color, high quality, highly detailed, elegant, sharp focus",
    "concept art, charachter concepts, digital painting, mystery, adventure",
    batch_size=3,
)

plot_images(images)

benchmark_result = []
start = time.time()
images = model.text_to_image(
    "A cute otter in a rainbow whirlpool holding shells, watercolor",
    batch_size = 3,
)

end = time.time()
benchmark_result.append(["Standard", end-start])
plot_images(images)

print(f"Standard model: {(end - start):.2f} seconds")
keras.backend.clear_session()

keras.mixed_precision.set_global_policy("mixed_float16")

model = keras_cv.models.StableDiffusion()

print("Compute dtype:", model.diffusion_model.compute_dtype)

print("Variable dtype:", model.diffusion_model.variable_dtype)



model.text_to_image("warning up the model", batch_size=3)

start = time.time()
images = model.text_to_image(
    "a cute magical flying dog,fantasy art",
    "golden color , high quality , highly detailed , elegant, sharp focus",
    "concept art, character concepts, digital painting, mystery, adventure ",
    batch_size = 3,
)

end = time.time()
benchmark_result.append(["Mixed precision", end - start])
plot_images(images)

print(f"Mixed precision model: {(end - start):.2f} seconds")
keras.backend.clear_session()


# xla compilation
