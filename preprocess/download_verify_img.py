"""
下载国税网验证码，保存到dataset/captcha-images
"""

import os
import asyncio
from pyppeteer import launch
import hashlib
import time
import re
import base64


class BrowserHandle:
    """浏览器操作"""

    def __init__(self, debug=True, browser_path='chrome-win32/chrome.exe'):
        self.url = 'https://inv-veri.chinatax.gov.cn/'
        self.debug = debug
        self.browser = None
        self.browser_path = browser_path
        if not self.browser_path:
            raise Exception('没有找到浏览器！！！')
        self._call_async_func(self._init)

    @staticmethod
    def _call_async_func(func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(func(*args, **kwargs))
        loop.run_until_complete(task)
        return task.result()

    def fill_basement(self, inv_code, inv_num, inv_date, inv_chk):
        return self._call_async_func(self._fill_basement, inv_code, inv_num, inv_date, inv_chk)

    def get_verify_code(self, max_wait_time=15, prev=None):
        return self._call_async_func(self._get_verify_code, max_wait_time=max_wait_time, prev=prev)

    def check(self, check_str, inv_unique_id, max_wait_time=10):
        return self._call_async_func(self._check, check_str, inv_unique_id, max_wait_time)

    def close(self):
        return self._call_async_func(self._close)

    async def _init(self):
        self.browser = await launch(headless=(not self.debug), ignoreHTTPSErrors=True, defaultViewport=None,
                                    # executablePath=self.browser_path,  # 'chrome-win32/chrome.exe',
                                    viewport={'width': 1920, 'height': 1080},
                                    args=['--disable-infobars',
                                          '--start-maximized',
                                          '--disable-dev-shm-usage',
                                          '--no-sandbox',
                                          '--disable-gpu',
                                          '--disable-extensions',
                                          '--disable-blink-features=AutomationControlled'])
        self.page = await self.browser.newPage()
        await self.page.evaluate(
            '''() =>{ Object.defineProperties(navigator,{ webdriver:{ get: () => false } }) }''')
        await self.page.goto(self.url)

    async def _fill_basement(self, inv_code, inv_num, inv_date, inv_chk):
        """填写基础信息"""
        fpdm = await self.page.waitForXPath('//*[@id="fpdm"]')
        await fpdm.click({'clickCount': 3})
        await fpdm.type(inv_code)
        fphm = await self.page.waitForXPath('//*[@id="fphm"]')
        await fphm.click({'clickCount': 3})
        await fphm.type(inv_num)
        kprq = await self.page.waitForXPath('//*[@id="kprq"]')
        await kprq.click({'clickCount': 3})
        await kprq.type(inv_date)
        kjje = await self.page.waitForXPath('//*[@id="kjje"]')
        await kjje.click({'clickCount': 3})
        await kjje.type(inv_chk)

    async def _get_verify_code(self, max_wait_time=15, prev=None):
        """获取验证码"""
        print('get verify code...')
        default_str = 'https://inv-veri.chinatax.gov.cn/images/code.png'
        prev_str = prev if prev else default_str
        if prev_str != default_str:
            ele_yzm_img = await self.page.waitForXPath('//*[@id="yzm_img"]')
            await ele_yzm_img.click()
        prev_str = hashlib.md5(prev_str.encode('utf-8')).hexdigest()
        cnt = 0
        ele_yzm = await self.page.waitForXPath('//*[@id="yzm_img"]')
        yzm_base64_str = await (await ele_yzm.getProperty("src")).jsonValue()
        curr_str = hashlib.md5(yzm_base64_str.encode('utf-8')).hexdigest()
        while cnt < max_wait_time and curr_str == prev_str:
            print(f'while loop:{cnt}\t{yzm_base64_str[:50]}')
            time.sleep(.5)
            ele_yzm = await self.page.waitForXPath('//*[@id="yzm_img"]')
            yzm_base64_str = await (await ele_yzm.getProperty("src")).jsonValue()
            curr_str = hashlib.md5(yzm_base64_str.encode('utf-8')).hexdigest()
            cnt += 1
        if prev_str == curr_str:
            print('No verify img!!!')
            return None, None
        ele_info = await self.page.waitForXPath('//*[@id="yzminfo"]')
        info = await (await ele_info.getProperty('textContent')).jsonValue()
        print(f'Tip:  {info}')
        return yzm_base64_str, info

    async def _close(self):
        if self.browser:
            await self.browser.close()

    def __del__(self):
        if self.browser:
            self._call_async_func(self.browser.close)


def base64_to_img(base64_str):
    """base64对象转换为图片"""
    print('保存结果...')
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    if not os.path.isdir('dataset/captcha-images'):
        os.makedirs('dataset/captcha-images')
    with open(f'dataset/captcha-images/{time.time()}.png', 'wb') as fb:
        fb.write(byte_data)


def task(idx):
    print(f'start {idx}...')
    inv_code, inv_num, inv_date, inv_chk = '033002000911', '69431322', '20210423', '681522'
    bh = BrowserHandle(debug=False)
    bh.fill_basement(inv_code, inv_num, inv_date, inv_chk)
    base64_img, info = bh.get_verify_code()
    base64_to_img(base64_img)
    with open('tips.txt', 'a') as fp:
        fp.write(info+'\n')


if __name__ == '__main__':
    cnt = 1
    while True:
        task(cnt)
        cnt += 1
