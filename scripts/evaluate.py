import torch
import torch.nn.functional as F

def predict_tile(model, input_tensor, class_labels=None, device="cpu"):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_class = probs.topk(1, dim=1)

    if class_labels is None:
        class_labels = [str(i) for i in range(probs.shape[1])]

    predicted_label = class_labels[top_class.item()]
    confidence = top_prob.item()

    return predicted_label, confidence, probs
