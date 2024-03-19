import torch

class TemperatureScaledModel(torch.nn.Module):
    def __init__(self, model, temperature=1.0):
        super(TemperatureScaledModel, self).__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, token_type_ids=None, attention_mask=attention_mask).logits
        # Scale logits by temperature
        scaled_logits = logits / self.temperature
        return scaled_logits