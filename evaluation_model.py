import sys, os, time
sys.dont_write_bytecode = True
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import config as cf
from multiprocessing import freeze_support

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    model_path = sys.argv[1] # モデルのパス
    dataset_path = sys.argv[2] # テスト用の画像が入ったディレクトリのパス

    # モデルの定義と読み込みおよび評価用のモードにセットする
    model = cf.build_model().to(DEVICE)
    if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
    else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
    model.eval()

    # データの読み込み (バッチサイズは適宜変更する)
    s_tm = time.time()
    data_transforms = T.Compose([T.Resize(cf.cellSize), T.CenterCrop(cf.cellSize), T.ToTensor()])
    test_data = ImageFolder(dataset_path, data_transforms)
    print(test_data.class_to_idx)
    bs = cf.batchSize
    # bs = int(bs * 1.2) # 必要メモリ量に応じた調整 (場合によっては1以下をかける)
    test_loader = DataLoader(test_data, batch_size = bs, num_workers = 0)

    label_list, pred_list = [], []
    for i, (data, label) in enumerate(test_loader):
        data = data.to(DEVICE)
        label = label.numpy().tolist()
        outputs = model(data)
        pred = torch.argmax(outputs, axis = 1).cpu().numpy().tolist()
        label_list += label
        pred_list += pred

        print(f"\r dataset loading: {i + 1} / {len(test_loader)}", end = "", flush = True)
    print()

    print(accuracy_score(label_list, pred_list)) # 正解率
    print(confusion_matrix(label_list, pred_list)) # 混同行列
    print(classification_report(label_list, pred_list)) # 各種評価指標
    print("done %.0fs" % (time.time() - s_tm))

if __name__ == '__main__':
    freeze_support()
    main()