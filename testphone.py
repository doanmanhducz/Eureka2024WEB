import asyncio
from truecallerpy import search_phonenumber

def check_phone_number(phone_number):
    # Kiểm tra nếu đầu vào không phải là số
    if not phone_number.isdigit():
        return "Số điện thoại không hợp lệ"

    country_code = "VN"
    installation_id = "a1i0I--jMM3uXFb-ofc-ODmqyAGq8gHtLFxVeOdmifPv9kJWNeNABir5r72aykMM"

    response = asyncio.run(search_phonenumber(phone_number, country_code, installation_id))

    try:
        name_value = response['data']['data'][0]['name']
        spam_score = response['data']['data'][0].get('spamScore', 0)
        spam_type = response['data']['data'][0].get('spamType', 'Unknown')
        
        if spam_score > 0:
            return f"Phishing Alert: {name_value} (Spam Score: {spam_score}, Spam Type: {spam_type})"
        else:
            return f"Name: {name_value}"
    except KeyError:
        return "Đây là SĐT Bình thường"
    except IndexError:
        return "Đây là SĐT Bình thường"
    except Exception as e:
        print("Exception:", e)
        return "Đã xảy ra lỗi khi xử lý yêu cầu"