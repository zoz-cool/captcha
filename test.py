import requests

if __name__ == '__main__':
    res = requests.post('http://localhost:8899/captcha-label',
                  files={'file': ('captcha_test.png', open('images-color/1622952867.692906.black.png', 'rb'))})
    print(res.json())
