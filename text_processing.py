import spacy
import re
from langdetect import detect
from transformers import pipeline,MBartTokenizer, MBartForConditionalGeneration



# Load spaCy models for English and French
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# Load the tokenizer and model explicitly
tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-50')
model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')

# Load Hugging Face Sentence Splitter
hf_sentence_splitter = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def clean_text(text):
    text = re.sub(r"[\n\t\r]+", " ", text)  # Remove newlines, tabs, carriage returns
    text = re.sub(r"\*\*|\>\s*", "", text)  # Remove Markdown formatting
    text = re.sub(r"\brapport\s+quotidien(?:\s+d'activité)?\s*:?", "", text, flags=re.IGNORECASE)  # Remove variations
    return text.strip().lower()  # Convert to lowercase


def segment_text(text):
    try:
        # Detect language: returns 'en' for English and 'fr' for French
        lang = detect(text)
    except Exception as e:
        # If language detection fails, default to English
        lang = 'en'

    # Choose the appropriate spaCy model based on the detected language
    if lang == 'fr':
        doc = nlp_fr(text)
    else:
        doc = nlp_en(text)

    # Split the text into individual sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # If spaCy fails (too few sentences), use Hugging Face as backup
    if len(sentences) <= 1:
        hf_result = hf_sentence_splitter(text, max_length=200, truncation=True)
        sentences = hf_result[0]["generated_text"].split("\n")

    return sentences