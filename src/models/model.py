from transformers import AutoModelForTokenClassification, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn

MODEL_CKPT =  "bert-base-uncased"
NUM_LABELS = 2

#model = AutoModelForTokenClassification(MODEL_CKPT, num_labels=2)
class SteamModel(nn.Module):
  
  def __init__(self,MODEL_CKPT,NUM_LABELS): 
    super(SteamModel,self).__init__() 
    self.num_labels = NUM_LABELS 

    #Load Model with given checkpoint and extract its body
    self.model = model = AutoModel.from_pretrained(MODEL_CKPT,config=AutoConfig.from_pretrained(MODEL_CKPT, output_attentions=True,output_hidden_states=True))
    self.dropout = nn.Dropout(0.2) 
    self.classifier = nn.Linear(768,NUM_LABELS) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)