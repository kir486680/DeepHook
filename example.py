from transformers import AutoTokenizer, AutoModelForCausalLM
from DeepHook import TraceMultiple 
import torch 

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def get_last_word(outputs_exp):# Assuming 'output' is the output of your model
    output_ids = outputs_exp.logits.argmax(-1)
    output_ids = output_ids[:, -1] # get the last predicted token of each batch
    decoded_output = tokenizer.decode(output_ids) # Now use the tokenizer to decode the output
    print(decoded_output)

text = "My name is"
encoded_input = tokenizer(text, return_tensors='pt')

def edit_fn1(output, layer_name):
    print('In the', layer_name)
    return output 

def edit_fn2(output, layer_name):
    print('In the', layer_name)


layers = {
    'transformer.wpe': (True, False, None),
    'transformer.h.0.mlp.c_fc': (True, True, edit_fn2),
    'transformer.h.0.mlp.c_proj': (True, True, edit_fn2)
}

with TraceMultiple(model, layers) as tm:
    outputs_exp = model(**encoded_input)
    get_last_word(outputs_exp)
    print(tm['transformer.wpe'].output)
    print(tm['transformer.h.0.mlp.c_fc'].output.shape == tm['transformer.h.0.mlp.c_proj'].input.shape)