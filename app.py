from flask import Flask, render_template, request
import asyncio
from truecallerpy import search_phonenumber
from langdetect import detect
import numpy as np
import re
from  nltk.tokenize import word_tokenize
import nltk
import textdistance
from nltk.corpus import stopwords
import pickle
from flask import Flask, request, jsonify
import predictor
from predictor import predict_text_recording, predict_recording
import testmail

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# with open('rf.pkl', 'rb') as file:
#     rf = pickle.load(file)

app = Flask(__name__)

# URLREGEX = r"^(https?|ftp)://[^\s/$.?#].[^\s]*$"
# URLREGEX_NOT_ALONE = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
# FLASH_LINKED_CONTENT = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])+).*\.swf"
# HREFREGEX = '<a\s*href=[\'|"](.*?)[\'"].*?\s*>'
# IPREGEX = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))\b"
# MALICIOUS_IP_URL = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\/(www|http|https|ftp))\b"
# EMAILREGEX = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
# GENERAL_SALUTATION = r'\b(dear|hello|Good|Greetings)(?:\W+\w+){0,6}?\W+(user|customer|seller|buyer|account holder)\b'


# def extract_feature(mail):
#     urls = findallurls(mail)
#     feature = [0] * (9)
#     i = 0
#     feature[i]= int(body_richness(mail))
#     i+=1
#     feature[i]= int(presenceGeneralSalutation(mail)==True)
#     i+=1
#     feature[i]= maliciousURL(urls)
#     i+=1
#     feature[i]= hexadecimalURL(urls)
#     i+=1
#     feature[i]= int(contains_prime_targets(mail)==True)
#     i+=1
#     feature[i]= int(contains_account(mail)==True)
#     i+=1
#     feature[i]= int(contains_suspended(mail)==True)
#     i+=1
#     feature[i] = int(contains_password(mail)==True)
#     i+=1
#     feature[i]= len(urls)
#     return feature

# def cleanBody(mail_body):
#         filtered = []
#         filtered_text = cleanpunc(mail_body)
#         word_tokens = word_tokenize(filtered_text)
#         for w in word_tokens:
#                 if w not in stop_words and w.isalpha():
#                     filtered.append(w)
#         return filtered

# def body_richness(mail_body):
#     mail_body = cleanBody(mail_body)
#     if len(set(mail_body))!=0:
#         return (len(mail_body)/len(set(mail_body)))
#     else:
#         return len(mail_body)

# def presenceGeneralSalutation(message):
#     return int(re.compile(GENERAL_SALUTATION,re.IGNORECASE).search(message) != None) == True

# def maliciousURL(urls):
#     count = 0
#     for url in urls:
#         if ((re.compile(IPREGEX, re.IGNORECASE).search(url)
#             is not None) == True or (len(re.compile(r'(https?://)',re.IGNORECASE).findall(url)) > 1)
#                 or (len(re.compile(r'(www.)',re.IGNORECASE).findall(url)) > 1)
#                 or (len(re.compile(r'(\.com|\.org|\.co)',re.IGNORECASE).findall(url)) > 1))== True:
#             count += 1
#     return count

# def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
#     cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
#     cleaned = re.sub(r'[.|,|)|(|\|/]',r'',cleaned)
#     return  cleaned

# def hexadecimalURL(urls):
#     count = 0
#     for url in urls:
#         if ((re.compile(r'%[0-9a-fA-F]+', re.IGNORECASE).search(url)
#             is not None) == True):
#             count += 1
#     return count

# def findallurls (content):
#     regex = r"(https?://\S+)"

#     urls = re.findall(regex, content)
#     return urls

# def contains_prime_targets(subject):
#     subject = purify(subject)
#     jaro = textdistance.Jaro()
#     for w in subject.split():

#         if ((jaro('bank',w)) >0.9 or (jaro('Paypal',w)) >0.9 or (jaro('ebay',w)) >0.9 or (jaro('amazon',w)) >0.9):
#             return 1
#     return 0

# def contains_account(subject):
#     subject = purify(subject)
#     jaro = textdistance.Jaro()
#     for w in subject.split():

#         if (jaro('account',w)) >0.9 or jaro('profile',w) >0.9 or jaro('handle',w) >0.9 :
#             return 1
#     return 0

# def contains_suspended(subject):
#     subject = purify(subject)
#     jaro = textdistance.Jaro()
#     for w in subject.split():

#         if (((jaro('closed',w)) or jaro('expiration',w))or jaro('suspended',w)) >0.9 or jaro('terminate',w) >0.9 or jaro('restricted',w) >0.9:
#             return 1
#     return 0

# def contains_password(subject):
#     subject = purify(subject)
#     jaro = textdistance.Jaro()
#     for w in subject.split():

#         if (jaro('password',w)) >0.9 or jaro('credential',w) > 0.9:
#             return 1
#     return 0

# def purify(subject):
#     filtered = ""
#     word_tokens = word_tokenize(subject)
#     for w in word_tokens:
#         if w not in stop_words and w.isalpha():
#                 w = stemmer.stem(w)
#                 filtered+=(lemmatizer.lemmatize(w))
#                 filtered+=" "
#     return filtered

@app.route('/')
def test_truecaller():
    return render_template('index.html')
    
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/project.html')
def project():
    return render_template('project.html')

@app.route('/blacklist.html')
def blacklist():
    return render_template('blacklist.html')

@app.route('/thongke.html')
def thongke():
    return render_template('thongke.html')

@app.route('/report.html')
def report():
    return render_template('report.html')

@app.route('/phishing.html')
def phishing():
    return render_template('phishing.html')

@app.route('/knowledge.html')
def knowledge():
    return render_template('knowledge.html')

@app.route('/quest.html')
def quest():
    return render_template('quest.html')

@app.route('/checkvishing.html', methods=['GET', 'POST'])
def checkvishing():
    if request.method == 'POST':
        if 'audio' not in request.files:
            app.logger.error('No file part in request')
            return jsonify({'result': 'No file part'}), 400

        file = request.files['audio']
        if file.filename == '':
            app.logger.error('No selected file')
            return jsonify({'result': 'No selected file'}), 400

        try:
            # Process the file for prediction
            file_byte = file.read()
            app.logger.debug(f'File content (first 20 bytes): {file_byte[:20]}')
            res = predictor.predict_recording(file_byte)
            app.logger.debug(f'Prediction result: {res}')

            if res == "error":
                return jsonify({'result': 'Could not understand audio'}), 200
            elif res == "gg-error":
                return jsonify({'result': 'Error with Google Speech Recognition service'}), 200
            elif res <= 0.5:
                return jsonify({'result': 'Not Vishing'}), 200
            return jsonify({'result': 'Vishing'}), 200
        except Exception as e:
            app.logger.error(f'Error processing file: {e}', exc_info=True)
            return jsonify({'result': 'Error processing file'}), 500

    return render_template('checkvishing.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    prediction = predictor.predict_text_recording(text)
    return jsonify({'prediction': prediction})

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    audio_bytes = file.read()
    prediction = predictor.predict_recording(audio_bytes)
    if prediction is None:
        return jsonify({'error': 'Error processing audio'}), 500
    return jsonify({'prediction': prediction})

@app.route('/service.html', methods=["POST", "GET"])
def service():
    # if request.method == "POST":
    #     try:
    #         phone = request.form.get("phone_number")  # Dùng .get để tránh KeyError nếu không có giá trị
    #         if phone:
    #             country_code = "VN"
    #             installation_id = "a1i0I--jMM3uXFb-ofc-ODmqyAGq8gHtLFxVeOdmifPv9kJWNeNABir5r72aykMM"

    #             response = asyncio.run(search_phonenumber(phone, country_code, installation_id))
    #             try:
    #                 name_value = response['data']['data'][0]['name']
    #                 if name_value:
    #                     print(name_value)
    #             except KeyError:
    #                 name_value = "Đây là SĐT Bình thường"
    #             return render_template('service.html', result1=name_value)
    #         else:
    #             # Nếu số điện thoại trống thì xử lý email
    #             mail = request.form.get("email")  # Kiểm tra email
    #             if not mail:
    #                 return render_template('service.html', error="Vui lòng nhập số điện thoại hoặc email")

    #             print(mail)
                
    #             detected_lang = detect(mail)
    #             if detected_lang == "en":
    #                 f_en = extract_feature(mail)
    #                 f_en_np = np.array([f_en])
    #                 predict = rf.predict(f_en_np)
    #                 print(predict)
    #             elif detected_lang == "vi":
    #                 inputs = ['vi: ' + mail]
    #                 outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=512)
    #                 output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #                 output = output[0]
    #                 f_vi = extract_feature(output[4:])
    #                 f_vi_np = np.array([f_vi])
    #                 predict = rf.predict(f_vi_np)
    #                 print(predict)
    #             else:
    #                 predict = [-1]
                
    #             if predict[0] == 1:
    #                 result = "Đây là Email Lừa đảo"
    #             elif predict[0] == 0:
    #                 result = "Đây là Email Bình thường"
    #             else:
    #                 result = "Ngôn ngữ không hợp lệ"
                
    #             return render_template('service.html', result2=result)
    #     except Exception as e:
    #         print(f"Lỗi: {e}")
    #         return render_template('service.html', error="Đã xảy ra lỗi khi xử lý yêu cầu")
    # else:
        return render_template('service.html')
    
@app.route('/testemail', methods=['POST'])
def test_email():
    email_text = request.form['emailText']
    result = testmail.process_email(email_text)
    return jsonify({'result2': result})

@app.route('/testemail.html', methods=['GET'])
def test_email_page():
    return render_template('testemail.html')

@app.route('/install.html')
def install():
    return render_template('install.html')


if __name__ == "__main__":
    app.run(debug=True)

#host='127.0.0.1', port=5000, debug=True






