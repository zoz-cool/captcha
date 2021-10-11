"""
国税网验证码识别
@author:yooongchun@foxmail.com
"""
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from model.predict import predict

limiter = Limiter(key_func=get_remote_address, default_limits=[
                  '5/second', '100/minute', '500/hour', '2000/day'])
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


class Response(BaseModel):
    """响应对象"""
    status_code: Optional[int] = 200
    message: Optional[str] = '请求成功'
    data: Optional[dict] = None


@app.post('/captcha/label', response_model=Response)
def get_captcha_label(request: Request, file: UploadFile = File(...)):
    """上传验证码图片返回识别标签"""
    try:
        img = Image.open(file.file)
        label = predict(img)
        return {'data': {'predict-label': label, 'filename': file.filename}}
    except Exception as e:
        return {'status_code': 500, 'message': '请求失败：'+str(e)}


def test():
    """测试用例"""
    import os
    from fastapi.testclient import TestClient
    client = TestClient(app)
    file_path = 'test_case/DRCMCC_1503.png'
    response = client.post('/captcha/label', files={
        'file': (os.path.basename(file_path), open(file_path, "rb"), "image/png")})
    print(response.json())


if __name__ == '__main__':
    test()
