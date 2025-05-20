import gradio as gr
from fastai.vision.all import load_learner, PILImage

# Load the trained model
learn = load_learner('mnist_model.pkl')

def predict(img):
    img = PILImage.create(img)
    pred_class, pred_idx, probs = learn.predict(img)

    # Only accept predictions of "3" or "7"
    if pred_class in ["3", "7"]:
        return f"✔️ Correct: {pred_class}, Confidence: {probs[pred_idx]:.4f}"
    else:
        return f"❌ Error: Model predicted {pred_class}, but only 3 or 7 are valid!"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(image_mode='L'),
    outputs="text",
    title="MNIST Classifier",
    description="Upload an image of a digit. Only 3 and 7 are valid.",
)

interface.launch()