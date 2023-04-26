import torch

class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fnV,
        loss_fnT,
        language_module,
        vision_module,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fcV = torch.nn.Linear(
            in_features=1000, 
            out_features=100
        )
        self.fcT = torch.nn.Linear(
            in_features=768, 
            out_features=100
        )
        self.classV = torch.nn.Linear(
            in_features=100, 
            out_features=num_classes
        )
        self.classT = torch.nn.Linear(
            in_features=100, 
            out_features=num_classes
        )
        self.loss_fnV = loss_fnV
        self.loss_fnT = loss_fnT
        self.dropoutV = torch.nn.Dropout(dropout_p)
        self.dropoutT = torch.nn.Dropout(dropout_p)
        
    def forward(self, image, input_ids, attention_mask, label=None):
        #_, pooled_output = self.language_module(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = self.language_module(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = pooled_output[0][:, 0, :].squeeze()
        text_features = torch.nn.functional.relu(
            pooled_output
        )
        image_features = torch.nn.functional.relu(
            self.vision_module(image)
        )
        image_features = self.dropoutV(self.fcV(image_features))
        text_features = self.dropoutT(self.fcT(text_features))
        predV = self.classV(image_features)
        predT = self.classT(text_features)
        predV = torch.nn.functional.softmax(predV)
        predT = torch.nn.functional.softmax(predT)
        lossV = (
            self.loss_fnV(predV, label)
            if label is not None else label
        )
        lossT = (
            self.loss_fnT(predT, label) 
            if label is not None else label
        )
        return (predV, predT, lossV, lossT)