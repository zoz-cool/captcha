import requests


if __name__ == '__main__':
    res = requests.post('http://www.yooongchun.com:8081/captcha-label',
                  files={'file': ('captcha_test.png', open('images/1622952867.692906.black.png', 'rb'))})
    print(res.json())
