from typing import Optional
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from model.predict import predict

app = FastAPI()


class Response(BaseModel):
    """响应对象"""
    status_code: Optional[int] = 200
    message: Optional[str] = '请求成功'
    data: Optional[dict] = None


@app.post('/captcha/label', response_model=Response)
def get_captcha_label(file: UploadFile = File(...)):
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
