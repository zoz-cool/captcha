from model.config import Config
from model.main import Model
from multiprocessing import Process, freeze_support
from visdom import server
import time

if __name__ == '__main__':
    # ================Train Model==================
    # 配置模型参数，也可以直接修改Config类
    config = Config()
    config.save_n_epoch = 1
    config.batch_size = 16
    config.model_path = 'model/model/checkpoint/captcha.ctc-model-3.pth'
    # 启动训练过程监测，通过浏览器打开http://localhost:8097查看
    # freeze_support()
    # Process(target=server.download_scripts_and_run).start()
    # time.sleep(2)
    # 开启模型训练
    model = Model()
    # model.fit()

    model.test()
    model.view()
