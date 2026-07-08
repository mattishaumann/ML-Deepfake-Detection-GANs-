# Detecting GAN-Generated Face Deepfakes

Can a model tell a real face from a GAN-generated one? Final exam paper for the Machine Learning and Deep Learning course, MSc Business Administration and Data Science (CBS). Written with Sven Kurth and Sebastian Uedingslohmann.

📄 **[Read the paper (PDF)](paper.pdf)** · 📓 **[The notebook](MLDL_GAN_generated_Face_Image_Detection.ipynb)**

## What we did

- Built a balanced dataset of 16,000 face images: 8,000 real (FFHQ, CelebA) and 8,000 synthetic from four GAN architectures (StyleGAN2, StyleGAN3, ProGAN, StarGAN)
- Compared three models: SVM with RBF kernel, a shallow CNN (2 conv layers), and a deeper CNN (4 conv layers)
- Tested the winner on a GAN type it had never seen (ProjectedGAN)

## What we found

- The deeper CNN won with **91% test accuracy**
- ProGAN fakes were nearly always caught; the more modern StyleGAN2/3 images were the hardest to spot
- On the unseen GAN, accuracy dropped to **59%** - detectors that work on known generators do not automatically generalize to new ones, which is exactly the problem platforms face in the wild

## Stack

Python, TensorFlow/Keras, scikit-learn, NumPy, Pandas, PIL/imagehash for dedup - all in one notebook.
