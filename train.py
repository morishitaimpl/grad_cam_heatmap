# macosでの学習時はnum_workersは0にしてください
import sys, time, os
sys.dont_write_bytecode = True
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import ImageFolder
import numpy as np
import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
id_str = sys.argv[1] # 識別用のID
dataset_path = sys.argv[2] # フォルダのあるパス
path_log = "_l_" + id_str + ".csv"
log_dir = "_log_" + id_str
if not os.path.exists(log_dir): os.mkdir(log_dir) # モデルの保存用のフォルダ
disp_score_t = ""

datasets_raw = ImageFolder(dataset_path, cf.data_transforms) # データの読み込み

# 学習・検証データを分割
train_data_size = int(cf.splitRateTrain * len(datasets_raw))
val_data_size = len(datasets_raw) - train_data_size
train_dataset, val_dataset = torch.utils.data.random_split(datasets_raw, [train_data_size, val_data_size])
print(datasets_raw.class_to_idx)
print(len(datasets_raw), train_data_size, val_data_size)

train_loader = DataLoader(train_dataset, batch_size = cf.batchSize, num_workers = 0, shuffle = True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size = cf.batchSize, num_workers = 0, drop_last=True)

# モデル、損失関数、最適化関数、収束率の定義
model = cf.build_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(), lr = 0.0005, momentum = 0.9)
rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 6, gamma = 0.1)

def Train_Eval(model, criterion, optimizer, scheduler, data_loader, device, epoch, max_epoch, is_val = False):
    total_loss = 0.0
    total_acc = 0.0
    counter = 0
    global disp_score_t
    model.eval() if is_val else model.train()
    
    # 検証データの最初のバッチのみGrad-CAMを可視化
    if is_val:
        first_batch = next(iter(data_loader))
        first_data, first_label = first_batch
        first_data = first_data.to(device)
        
        # 予測クラスを取得
        with torch.no_grad():
            output = model(first_data[0:1])
            pred_class = output.argmax(dim=1).item()
            pred_prob = torch.softmax(output, dim=1)[0][pred_class].item()
        
        # 各層でGrad-CAMを可視化
        target_layers = [
            model.model_pre.features[-1],  # EfficientNetV2の最後の畳み込み層
            model.conv1,  # 追加の畳み込み層1
            model.conv2   # 追加の畳み込み層2
        ]
        
        for i, target_layer in enumerate(target_layers):
            layer_name = f"layer_{i+1}"
            cf.visualize_gradcam(
                model, 
                first_data[0:1], 
                target_layer,
                target_class=pred_class,
                save_path=f"{log_dir}/gradcam_{id_str}_{epoch + 1:03}_{layer_name}_class{pred_class}_prob{pred_prob:.3f}.png"
            )
    
    for n, (data, label) in enumerate(data_loader): # バッチ毎にデータ読み込み
        counter += data.shape[0]
        if is_val == False: optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        if is_val:
            with torch.no_grad():
                output = model(data)
        else:
            output = model(data)
        
        loss = criterion(output,label)
        total_loss += loss.item()
        total_acc += cf.calc_acc(output, label)
        # print(loss)

        if is_val == False:
            loss.backward()
            optimizer.step()
        # 1, 2 学習データに対するloss, acuuracy
        # 3, 4 検証データに対するloss, accuracy
        # lossおよび精度を表示・1エポックが終わる度にかかった時間も表示する

        if is_val == False:
            disp_score_t = f"{epoch + 1:03} / {max_epoch:03} [ {n + 1:04} / {len(data_loader):04} ] l: {total_loss/(n+1):.05f} a: {total_acc/counter:.03f}"
            print(f"\r {disp_score_t}", end = "")
        else: 
            print(f"\r {disp_score_t} l: {total_loss / (n + 1):.05f} a: {total_acc / counter:.03f}", end = "")

    if is_val == False: scheduler.step() # 学習率の変更

    # 学習に使った画像の一部を保存
    torchvision.utils.save_image(data[:min(cf.batchSize, 16)], f"{log_dir}/_i_{id_str}_{epoch + 1:03}.png", value_range=(-1.0,1.0), normalize=True)

    return total_loss/(n+1), total_acc/counter

best_loss = None
with open(path_log, mode = "w") as f: f.write("")
s_tm = time.time()

for epoch in range(cf.epochSize):
    n_tm = time.time()
    train_loss, train_acc = Train_Eval(model, criterion, optimizer, rate_scheduler, train_loader, DEVICE, epoch, cf.epochSize) 
    val_loss, val_acc = Train_Eval(model, criterion, optimizer, rate_scheduler, val_loader, DEVICE, epoch, cf.epochSize, is_val = True)
    print(f" {time.time() - n_tm:.0f}s")

    # if best_loss is None or val_loss < best_loss: # 評価での誤差がよき感じでモデルを保存
    #     best_loss = val_loss
    torch.save(model.state_dict(), f"{log_dir}/_m_{id_str}_{epoch + 1:03}.pth") # モデルの保存

    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: f.write(f"{train_loss},{val_loss},{train_acc},{val_acc}\n")

# モデルの保存
torch.save(model.state_dict(), f"{log_dir}/_m_{id_str}_{cf.epochSize:03}.pth")
print(f"done {time.time() - s_tm:.0f}s")