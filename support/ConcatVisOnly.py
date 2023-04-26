import torch

class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        vision_module,
        vision_feature_dim,
        dropout_p,
        
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.vision_module = vision_module
        self.fc = torch.nn.Linear(
            in_features=1000, 
            out_features=num_classes
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, text, image, label=None):
        image_features = self.dropout(
            torch.nn.functional.relu(
                self.vision_module(image)
            )
        )
        logits = self.fc(image_features)
        pred = torch.nn.functional.softmax(logits)
        loss = (
            self.loss_fn(pred, label) 
            if label is not None else label
        )
        return (pred, loss)