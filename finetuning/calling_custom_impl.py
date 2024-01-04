from finetuning.cti.transformers.transformers.src.transformers.models.auto.modeling_auto import AutoModelForCausalLM as AutoModelForCausalLM

from finetuning.cti.transformers.transformers.src.transformers.models.auto.tokenization_auto import AutoTokenizer as AutoTokenizer

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-64g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

print("\n\n*** Generate:")

prompt_template = "Tell me about AI please"

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))