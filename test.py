import requests


if __name__ == '__main__':
    res = requests.post('http://captcha:5000/captcha-label',
                  files={'file': ('captcha_test.png', open('images/1622952867.692906.black.png', 'rb'))})
    if res.status_code == 200:
        print(res.json())
    else:
        print('错误码：', res.status_code)

