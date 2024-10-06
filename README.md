# Educational Chatbot Using Llama 2

## Overview

This project utilizes the Llama 2 chatbot technology to deliver responses to educational questions. It also recommends relevant articles based on keyword analysis of the chatbot responses, leveraging advanced techniques from natural language processing (NLP).

## Table of Contents

- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Tokenizer](#model-and-tokenizer)
- [Keyword Extraction](#keyword-extraction)
- [Article Recommendation](#article-recommendation)
- [License](#license)

## Technologies Used

- **Llama 2**: A state-of-the-art language model for generating conversational responses.
- **Transformers**: A library for state-of-the-art NLP.
- **KeyBERT**: For extracting relevant keywords from text.
- **BeautifulSoup**: For web scraping to find relevant articles.
- **Torch**: A deep learning framework that supports dynamic computation graphs.
- **Accelerate**: For optimizing model inference.

## Installation

To set up the project, install the required libraries using pip. Run the following commands:

```bash
pip install transformers torch accelerate
pip install KeyBERT
pip install beautifulsoup4 requests
```

### Hugging Face Login

To access the Llama 2 model, you need to authenticate your Hugging Face account. Use the following commands to log in:

```bash
!huggingface-cli login
!huggingface-cli whoami
```

## Usage

1. **Load the Model and Tokenizer**: The Llama 2 7B model is loaded using the Hugging Face Transformers library.

2. **Generate Chatbot Responses**: The `get_llama_response` function takes a user prompt and generates a response using the Llama 2 model.

3. **Keyword Extraction**: Using KeyBERT, important keywords from the chatbot's response are extracted to recommend relevant articles.

4. **Web Scraping for Articles**: The program scrapes a target website to find articles related to the extracted keywords.

### Example

```python
prompt = 'Can you explain what dark matter is?'
response = get_llama_response(prompt)
print("Chatbot:", response)
```

## Model and Tokenizer

The model is loaded with the following code:

```python
from transformers import AutoTokenizer
import torch

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### Notes on Model Usage

- The model uses 16-bit floating point format (float16) to reduce memory usage during inference.
- The `device_map="auto"` option allows the pipeline to automatically select the appropriate device (CPU or GPU).

## Keyword Extraction

The KeyBERT library is used to extract the top 10 keywords from the chatbot response:

```python
from keybert import KeyBERT

model = KeyBERT(model="distilbert-base-nli-mean-tokens")

keywords = model.extract_keywords(
    response,
    top_n=10,
    keyphrase_ngram_range=(1, 1),
    stop_words="english",
)
```

## Article Recommendation

Using BeautifulSoup, relevant articles are scraped based on the extracted keywords. The function `extract_articles_with_urls` checks for keyword matches in the article titles:

```python
import requests
from bs4 import BeautifulSoup

base_url = "https://www.livescience.com/"

def extract_articles_with_urls(soup, keywords):
    articles_with_urls = []
    for heading in soup.find_all("h3", class_="article-name"):
        title = heading.text.strip()
        if any(keyword.lower() in title.lower() for keyword in keywords):
            article_url = heading.find_parent("a")["href"]
            articles_with_urls.append((title, article_url))
    return articles_with_urls
```

### Example of Output

Upon running the script, the program will display the recommended articles based on the keywords extracted from the chatbot's response.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
