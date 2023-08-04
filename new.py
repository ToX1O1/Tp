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
    prompt_text: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"Fuck": "You"}


prompt_prefix = "### Generate an intelligence report after analysing the following text for inferences of any actionable crime intelligence, remember and include as many salient details and specifics as possible. Explain your thinking for all your analysis with evidence from the text. Be verbose: '"
prompt_suffix = "' ### The Crime Intelligence report for this text is: "

def create_batches(prompt):
    global llm
    batches = []
    appendage_tokens = llm.tokenize((prompt_prefix + prompt_suffix).encode('utf-8'))
    tokens = llm.tokenize(prompt.encode('utf-8'))
    tokenlen = len(tokens)
    max_batchlen = CONTEXT_WINDOW - len(appendage_tokens)
    if tokenlen >= max_batchlen:
        counter = 0
        while counter < tokenlen:
            upper_end = counter + max_batchlen
            if upper_end > tokenlen:
                batch = tokens[counter:]
            else:
                batch = tokens[counter:upper_end]
            batches.append(batch)
            counter += max_batchlen

    return batches


@app.post("/prompt")
def send_prompt(jsondata: JSONInput):
    global llm
    prompt_text = jsondata.prompt_text
    tokens = llm.tokenize(prompt_text.encode('utf-8'))
    tokenlen = len(tokens)
    print(f'Prompt received : Token Length = {tokenlen}')
    if tokenlen >= 2048:
        report = ""
        token_batches = create_batches(prompt_text)
        for token_batch in token_batches:
            batch_text = llm.detokenize(token_batch).decode("utf-8")
            final_prompt = prompt_prefix + batch_text + prompt_suffix
            output = llm(str(final_prompt), max_tokens=2048)
            output_str = output['choices'][0]['text']
            report += "\n" + output_str
        return report
    else: 
        final_prompt = prompt_prefix + prompt_text + prompt_suffix
        output = llm(str(final_prompt), max_tokens=2048)
        output_str = output['choices'][0]['text']
    return output_str