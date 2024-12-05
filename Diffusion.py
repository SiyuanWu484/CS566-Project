import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), _ = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0  # Normalize to [0, 1]
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension

# Build denoising network
def build_denoiser(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # First convolutional block
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        
        # Additional convolutional block
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        
        # Back to lower number of filters for reconstruction
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same')  # Output layer without activation
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# Parameters
epochs = 10
batch_size = 64
steps = 50  # Number of forward diffusion steps
noise_level = 0.1
num_samples = 100  # Number of images to generate (10×10 grid)

# Build and train the denoising model
denoiser = build_denoiser(input_shape=(28, 28, 1))

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i + batch_size]

        # Forward diffusion: Add noise step by step
        noised_images = batch
        for _ in range(steps):
            noise = np.random.normal(0, noise_level, noised_images.shape)
            noised_images = np.clip(noised_images + noise, 0, 1)

        # Train denoiser to reconstruct the original images
        denoiser.train_on_batch(noised_images, batch)

    # Generate samples for visualization
    sampled_images = []
    for _ in range(num_samples):
        test_image = np.expand_dims(X_train[np.random.randint(len(X_train))], axis=0)
        noised_image = test_image
        for _ in range(steps):
            noise = np.random.normal(0, noise_level, noised_image.shape)
            noised_image = np.clip(noised_image + noise, 0, 1)
        denoised_image = denoiser.predict(noised_image)
        sampled_images.append(np.clip(denoised_image[0], 0, 1))  # Ensure valid pixel range

    # Plot 10×10 grid of generated images
    plt.figure(figsize=(10, 10))
    for idx, img in enumerate(sampled_images):
        plt.subplot(10, 10, idx + 1)
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Generated Images at Epoch {epoch + 1}", fontsize=16)
    plt.tight_layout()
    plt.show()
