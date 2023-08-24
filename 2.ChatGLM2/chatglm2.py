#在终端运行，设置export GLOG_v=3，去掉warning

import os
import platform
import signal
import readline

import time
import mindspore as ms
import numpy as np
from mindformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("glm2_6b")
model = AutoModel.from_pretrained("glm2_6b")

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0

        inputs = tokenizer(query)
        outputs = model.generate(np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
                                 max_length=100, do_sample=False, top_p=0.7, top_k=1)
        response = tokenizer.decode(outputs)[0]
        history = history + [(query, response)]

        if stop_stream:
            stop_stream = False
            break
        else:
            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(build_prompt(history), flush=True)
                signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()