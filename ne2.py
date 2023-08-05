from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from queue import Queue
from llama_cpp import Llama
from pymongo import MongoClient
from bson.objectid import ObjectId
import threading

#Eg. url_id -> 64cd40add9dc8c347abd1443


class REQObject(BaseModel):
    obj_id: str
    body: str


llama_q = Queue()


MONGO_URI = "mongodb+srv://yash23malode:9dtb8MGh5aCZ5KHN@cluster.u0gqrzk.mongodb.net/"
DB_NAME = "prakat23"
COLL_NAME = "crawled_sites"


mongoclient = MongoClient(MONGO_URI)
db = mongoclient[DB_NAME]
collection = db[COLL_NAME]
report_collection = db["report_collection"]


CONTEXT_WINDOW = 2048

llm = Llama(
    model_path="Wizard-Vicuna-7B-Uncensored.ggmlv3.q8_0.bin",
    n_gpu_layers=64,  # Set the number of layers to run on the GPU
    n_ctx=CONTEXT_WINDOW
)


prompt_prefix = "### Generate an intelligence report after analysing the following text for inferences of any actionable crime intelligence, remember and include as many salient details and specifics as possible. Explain your thinking for all your analysis with evidence from the text. Be verbose: '"

prompt_suffix = "' ### The Crime Intelligence report for this text is: "


def create_batches(prompt, prompt_prefix, prompt_suffix):
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


def send_prompt(prompt_text):
    global llm
    global prompt_prefix
    global prompt_suffix
    total_prompt = prompt_prefix + prompt_text + prompt_suffix
    tokens = llm.tokenize(total_prompt.encode('utf-8'))
    tokenlen = len(tokens)
    print(f'Prompt received : Token Length = {tokenlen}')
    if tokenlen >= 2048:
        report = ""
        token_batches = create_batches(prompt_text, prompt_prefix, prompt_suffix)
        for token_batch in token_batches:
            batch_text = llm.detokenize(token_batch).decode("utf-8")
            final_prompt = prompt_prefix + batch_text + prompt_suffix
            output = llm(str(final_prompt), max_tokens=2048)
            output_str = output['choices'][0]['text']
            report += "\n" + output_str
        return report.strip()
    else: 
        output = llm(str(total_prompt), max_tokens=2048)
        output_str = output['choices'][0]['text']
    return output_str


def worker():
    while True:
        req = llama_q.get()
        if req is None:
            continue
        req_id = req.obj_id
        req_body = req.body
        output = send_prompt(body)
        output_doc = {
            "url_id": req_id,
            "report": output
        }
        report_collection.insert_one(output_doc)
        collection.update({"_id": ObjectId(req_id)}, {"$set": {"report_generated": True}})


llama_daemon = threading.Thread(target=worker, daemon=True)
# llama_daemon.start()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Fuck": "You"}


@app.get("/genreport/{url_id}")
def generate_prompt(url_id: str):
    document = collection.find_one({"_id": ObjectId(url_id)})
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    body_text = document["body"]
    request_object = REQObject(obj_id=url_id, body=body_text)
    llama_q.put(request_object)
    return "REPORT REQUESTED for id:{url_id}"
    








# if __name__ == "__main__":
#     #doc1 = collection.find_one({"_id":"64cd40add9dc8c347abd1443"})
#     doc1 = collection.find_one()
#     print(doc1)
