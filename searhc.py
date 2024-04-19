from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(tavily_api_key= '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')

#print(search.invoke("what is the weather in SF"))

from gpt4all import Embed4All
text = 'The quick brown fox jumps over the lazy dog'
embedder = Embed4All()
output = embedder.embed(text)
print(output)

"""for embeddings pip install sentence-transformers
for transformers pip install transformers

for cpu only pip install 'transformers[torch]'



for run offline set environment variable
Add 🤗 Datasets to your offline training workflow with the environment variable HF_DATASETS_OFFLINE=1.

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...

"""

"""hugginface testing"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
model = AutoModelForTokenClassification.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)