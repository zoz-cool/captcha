"""
下载国税网验证码，保存到dataset/captcha-images
"""

import asyncio
import base64
import hashlib
import os
import pathlib
import re
import time

import pandas as pd
from pyppeteer import launch

root_dir = pathlib.Path(__file__).parent.parent.parent


class BrowserHandle:
    """浏览器操作"""

    def __init__(self, debug=True):
        self.url = 'https://inv-veri.chinatax.gov.cn/'
        self.debug = debug
        self.browser = None
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

    def close(self):
        return self._call_async_func(self._close)

    async def _init(self):
        self.browser = await launch(headless=(not self.debug), ignoreHTTPSErrors=True, defaultViewport=None,
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
        """get verify code"""
        print('get verify code...')
        default_str = 'https://inv-veri.chinatax.gov.cn/images/code.png'
        prev_str = prev if prev else default_str
        if prev_str != default_str:
            await self.page.evaluate("()=>{document.querySelector('#yzm_img').click()}")
        prev_str = hashlib.md5(prev_str.encode('utf-8')).hexdigest()
        cnt = 0
        ele_yzm = await self.page.waitForXPath('//*[@id="yzm_img"]')
        yzm_base64_str = await (await ele_yzm.getProperty("src")).jsonValue()
        curr_str = hashlib.md5(yzm_base64_str.encode('utf-8')).hexdigest()        
        while cnt < max_wait_time and curr_str == prev_str:
            time.sleep(1)
            print(f'[{cnt}/{max_wait_time}]s wait verify image...')
            ele_yzm = await self.page.waitForXPath('//*[@id="yzm_img"]')
            yzm_base64_str = await (await ele_yzm.getProperty("src")).jsonValue()
            prev_str = curr_str
            curr_str = hashlib.md5(yzm_base64_str.encode('utf-8')).hexdigest()
            cnt += 1
        if prev_str == curr_str:
            print('ERROR!!!Can not get verify image!')
            return None, None
        ele_info = await self.page.waitForXPath('//*[@id="yzminfo"]')
        info = await (await ele_info.getProperty('textContent')).jsonValue()
        return yzm_base64_str, info

    async def _close(self):
        if self.browser:
            await self.browser.close()

    def __del__(self):
        if self.browser:
            self._call_async_func(self.browser.close)


def base64_to_img(base64_str, output_dir):
    """base64对象转换为图片"""
    print('保存结果...')
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/{int(time.time()*1000)}.png', 'wb') as fb:
        fb.write(byte_data)


def task(idx, inv_code, inv_num, inv_date, inv_chk, output_dir):
    print(f'[{idx}] check inv_code={inv_code}, inv_num={inv_num}...')
    bh = BrowserHandle(debug=False)
    bh.fill_basement(inv_code, inv_num, inv_date, inv_chk)
    prev_str = None
    for i in range(100):
        print(f'inner loop {i+1}')
        base64_img, info = bh.get_verify_code(prev=prev_str)
        if not info:
            break
        prev_str = base64_img
        base64_to_img(base64_img, output_dir)
        print(f'Tip message:{info}')


if __name__ == '__main__':
    inv_data = pd.read_csv(root_dir / 'assets/inv_data.csv', dtype=str)
    target_count = 100
    output_dir = root_dir / 'dataset/origin'
    for i in range(target_count):
        for _, row in inv_data.iterrows():
            if pd.isna(row['发票代码']):
                continue
            task(i+1, row['发票代码'], row['发票号码'], row['开票日期'], row['验证码'], output_dir)
