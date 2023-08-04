from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama


CONTEXT_WINDOW = 2048

llm = Llama(
    model_path="Wizard-Vicuna-7B-Uncensored.ggmlv3.q8_0.bin",
    n_gpu_layers=64,  # Set the number of layers to run on the GPU
    n_ctx=CONTEXT_WINDOW
)

class JSONInput(BaseModel):
    text: str
    sys_prompt: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prompt")
def send_prompt(jsondata: JSONInput):
    print(f'TEXT: {jsondata.text}')
    print(f'SYS_PROMPT: {jsondata.sys_prompt}')
    input_prompt = "Q: " + jsondata.text + " A: "
    end_prompt = jsondata.sys_prompt + input_prompt
    tokens = llm.tokenize(end_prompt.encode('utf-8'))
    tokenlen = len(tokens)
    if tokenlen >= CONTEXT_WINDOW:
        print(f'ERROR: Given input of tokens {tokenlen} is larger than Context Window {CONTEXT_WINDOW}')
        return ''
    output = llm(str(end_prompt), max_tokens=2048)
    output_str = output['choices'][0]['text']
    print(output_str)
    return output_str

