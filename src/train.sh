set -x

cd PaddleOCR
mkdir -p dataset
wget https://yooongchun-share.oss-cn-beijing.aliyuncs.com/dataset/captcha/captcha_dataset_v2.tgz && tar xf captcha_dataset_v2.tgz -C dataset

pip3 install https://paddle-wheel.bj.bcebos.com/2.4.0-rc0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.4.0rc0.post112-cp37-cp37m-linux_x86_64.whl
pip3 install -r requirements.txt

mkdir -p pretrained_model
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar -O pretrained_model/model.tar
cd pretrained_model && tar xf model.tar && cd ..

pretrained_model_path=./pretrained_model/pretrained_model/ch_PP-OCRv3_rec_train/best_accuracy
fleetrun --ips=$TRAINER_IP_LIST --selected_gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model=$pretrained_model_path

sleep 10d
