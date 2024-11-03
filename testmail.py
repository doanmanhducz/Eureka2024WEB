import pickle
import numpy as np
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import nltk
import string
import textdistance

# Initialize language processing tools
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# Initialize translation model
model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the pre-trained model
with open('rf.pkl', 'rb') as file:
    rf = pickle.load(file)

# Define regex patterns
URLREGEX = r"^(https?|ftp)://[^\s/$.?#].[^\s]*$"
URLREGEX_NOT_ALONE = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
FLASH_LINKED_CONTENT = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])+).*\.swf"
HREFREGEX = r'<a\s*href=[\'|"](.*?)[\'"].*?\s*>'
IPREGEX = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))\b"
MALICIOUS_IP_URL = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\/(www|http|https|ftp))\b"
EMAILREGEX = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
GENERAL_SALUTATION = r'\b(dear|hello|Good|Greetings)(?:\W+\w+){0,6}?\W+(user|customer|seller|buyer|account holder)\b'

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def cleanpunc(sentence):
    # Remove punctuation from the text
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r'', cleaned)
    return cleaned

def cleanBody(mail_body):
    filtered = []
    filtered_text = cleanpunc(mail_body)
    word_tokens = word_tokenize(filtered_text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w)
    return filtered

def body_richness(mail_body):
    mail_body = cleanBody(mail_body)
    if isinstance(mail_body, list):
        mail_body = ' '.join(mail_body)
    if len(set(mail_body.split())) != 0:
        return len(mail_body.split()) / len(set(mail_body.split()))
    else:
        return len(mail_body.split())

def presenceGeneralSalutation(mail):
    return bool(re.search(GENERAL_SALUTATION, mail))

def maliciousURL(urls):
    count = 0
    for url in urls:
        if ((re.compile(IPREGEX, re.IGNORECASE).search(url) is not None) == True or
            (len(re.compile(r'(https?://)', re.IGNORECASE).findall(url)) > 1) or
            (len(re.compile(r'(www.)', re.IGNORECASE).findall(url)) > 1) or
            (len(re.compile(r'(\.com|\.org|\.co)', re.IGNORECASE).findall(url)) > 1)) == True:
            count += 1
    return count

def hexadecimalURL(urls):
    count = 0
    for url in urls:
        if ((re.compile(r'%[0-9a-fA-F]+', re.IGNORECASE).search(url) is not None) == True):
            count += 1
    return count

def findallurls(content):
    regex = r"(https?://\S+)"
    urls = re.findall(regex, content)
    return urls

def purify(subject):
    filtered = ""
    word_tokens = word_tokenize(subject)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            w = stemmer.stem(w)
            filtered += lemmatizer.lemmatize(w)
            filtered += " "
    return filtered

def contains_prime_targets(subject):
    subject = purify(subject)
    jaro = textdistance.Jaro()
    for w in subject.split():
        if ((jaro('bank', w)) > 0.9 or (jaro('Paypal', w)) > 0.9 or (jaro('ebay', w)) > 0.9 or (jaro('amazon', w)) > 0.9):
            return 1
    return 0

def contains_account(subject):
    subject = purify(subject)
    jaro = textdistance.Jaro()
    for w in subject.split():
        if (jaro('account', w)) > 0.9 or jaro('profile', w) > 0.9 or jaro('handle', w) > 0.9:
            return 1
    return 0

def contains_suspended(subject):
    subject = purify(subject)
    jaro = textdistance.Jaro()
    for w in subject.split():
        if (((jaro('closed', w)) or jaro('expiration', w)) or jaro('suspended', w)) > 0.9 or jaro('terminate', w) > 0.9 or jaro('restricted', w) > 0.9:
            return 1
    return 0

def contains_password(subject):
    subject = purify(subject)
    jaro = textdistance.Jaro()
    for w in subject.split():
        if (jaro('password', w)) > 0.9 or jaro('credential', w) > 0.9:
            return 1
    return 0

def extract_feature(mail):
    if isinstance(mail, list):
        mail = ' '.join(mail)
    urls = findallurls(mail)
    feature = [0] * 9
    i = 0
    feature[i] = int(body_richness(mail))
    i += 1
    feature[i] = int(presenceGeneralSalutation(mail) == True)
    i += 1
    feature[i] = maliciousURL(urls)
    i += 1
    feature[i] = contains_prime_targets(mail)
    i += 1
    feature[i] = contains_account(mail)
    i += 1
    feature[i] = contains_suspended(mail)
    i += 1
    feature[i] = contains_password(mail)
    i += 1
    feature[i] = len(urls)
    return feature

def process_email(mail):
    try:
        print(mail)
        
        detected_lang = detect(mail)
        if detected_lang == "en":
            f_en = extract_feature(mail)
            f_en_np = np.array([f_en])
            predict = rf.predict(f_en_np)
            print(predict)
        elif detected_lang == "vi":
            inputs = ['vi: ' + mail]
            outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=512)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output = output[0]
            f_vi = extract_feature(output[4:])
            f_vi_np = np.array([f_vi])
            predict = rf.predict(f_vi_np)
            print(predict)
        else:
            predict = [-1]
        
        if predict[0] == 1:
            result = "Đây là Email Lừa đảo"
        elif predict[0] == 0:
            result = "Đây là Email Bình thường"
        else:
            result = "Ngôn ngữ không hợp lệ"
        
        return result
    except Exception as e:
        print(f"Lỗi: {e}")
        return "Đã xảy ra lỗi khi xử lý yêu cầu"