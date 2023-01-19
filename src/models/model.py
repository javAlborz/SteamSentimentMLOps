#from transformers import AutoModelForTokenClassification
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn


from transformers import PreTrainedModel, PretrainedConfig


class SteamConfig(PretrainedConfig):
    """
    SteamConfig class

    Extends PretrainedConfig class
    """

    model_type = 'SteamModel'

    def __init__(self, important_param=42, **kwargs):
        super().__init__(**kwargs)
        self.important_param = important_param


class SteamModel(PreTrainedModel):
    """
    SteamModel class

    Extends PreTrainedModel class
    """

    config_class = SteamConfig

    def __init__(self, config, MODEL_CKPT, NUM_LABELS):
        """
        Setup model

        Args:
            config (_type_): _description_
            MODEL_CKPT (_type_): _description_
            NUM_LABELS (_type_): _description_
        """
        super(SteamModel, self).__init__(config)
        self.config = config
        self.num_labels = NUM_LABELS

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(MODEL_CKPT, config=AutoConfig.from_pretrained(
            MODEL_CKPT, output_attentions=True, output_hidden_states=True))
        self.dropout = nn.Dropout(0.2)
        # load and initialize weights
        self.classifier = nn.Linear(768, NUM_LABELS)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)

        # Add custom layers
        # outputs[0]=last hidden state
        sequence_output = self.dropout(outputs[0])

        logits = self.classifier(
            sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)
