import requests

if __name__ == '__main__':
    res = requests.post('http://localhost:5000/captcha-label',
                  files={'file': ('captcha_test.png', open('dataset/captcha/test/2FA_39.png', 'rb'))})
    print(res.json())
