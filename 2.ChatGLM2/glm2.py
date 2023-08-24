#在终端运行，设置export GLOG_v=3，去掉warning

from mindformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("glm2_6b")
model = AutoModel.from_pretrained("glm2_6b")

query = "你好"

prompted_inputs = tokenizer.build_prompt(query)
input_tokens = tokenizer([prompted_inputs])

outputs = model.generate(input_tokens["input_ids"], max_length=100)
response = tokenizer.decode(outputs)[0]
print(response)