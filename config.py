import sys
sys.dont_write_bytecode = True
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 384

# クラス全体の数
classesSize = 3

# 繰り返す回数
epochSize = 15

# ミニバッチのサイズ
batchSize = 5

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8


# データ変換
data_transforms = T.Compose([
    T.Resize(int(cellSize * 1.2)), #リサイズ
    T.RandomRotation(degrees = 15), #回転
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5), #ぼかしの確率
    T.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = [-0.2, 0.2]), #色調整
    T.RandomHorizontalFlip(0.5), #左右反転
    T.RandomEqualize(p = 0.5), #ヒストグラム平坦化
    T.CenterCrop(cellSize), #中央切り抜き
    T.ToTensor(),
    # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)) #ランダムな大きさの矩形で画像をマスクする
    ])

def calc_acc(output, label): # 結果が一致するラベルの数をカウントする
    p_arg = torch.argmax(output, dim = 1)
    return torch.sum(label == p_arg)

class build_model(nn.Module):
    def __init__(self):
        super(build_model, self).__init__()
        self.model_pre = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)
        
        # 追加の畳み込み層
        self.conv1 = nn.Conv2d(1000, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, classesSize)
        
        # 特徴マップを保存するためのフック
        self.activations = {}
        
        # 各層にフックを登録
        self.model_pre.features.register_forward_hook(self.get_activation('features'))
        self.conv1.register_forward_hook(self.get_activation('conv1'))
        self.conv2.register_forward_hook(self.get_activation('conv2'))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def forward(self, input):
        # 特徴マップをクリア
        self.activations = {}
        
        # EfficientNetV2を通す
        x = self.model_pre(input)
        
        # 形状を調整して畳み込み層に通す
        x = x.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1000, 1, 1)
        
        # 追加の畳み込み層を通す
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 形状を調整
        x = x.squeeze(-1).squeeze(-1)  # (batch_size, 256)
        
        # ドロップアウトと分類層
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def visualize_feature_maps(model, input_tensor, layer_name, save_path=None):
    """
    指定した層の特徴マップを可視化する
    
    Args:
        model: モデル
        input_tensor: 入力テンソル
        layer_name: 可視化したい層の名前
        save_path: 保存先のパス（Noneの場合は表示のみ）
    """
    # モデルを評価モードに
    model.eval()
    
    # 推論を実行して特徴マップを取得
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 指定した層の特徴マップを取得
    feature_maps = model.activations[layer_name]
    
    # バッチサイズとチャンネル数を取得
    batch_size, channels, height, width = feature_maps.shape
    
    # 特徴マップを可視化
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.ravel()
    
    # 最初の16チャンネルを表示
    for i in range(min(16, channels)):
        feature_map = feature_maps[0, i].cpu().numpy()
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 勾配を保存するためのフック
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class=None):
        # モデルを評価モードに
        self.model.eval()
        
        # 推論を実行
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # ターゲットクラスに対する勾配を計算
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 勾配の平均を計算
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # 勾配の重みを計算
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 特徴マップの重み付け和を計算
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLUを適用して負の値を除去
        cam = F.relu(cam)
        
        # 正規化
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze().numpy()

def visualize_gradcam(model, input_tensor, target_layer, target_class=None, save_path=None):
    """
    Grad-CAMを可視化し、ヒートマップと元画像を合成する
    
    Args:
        model: モデル
        input_tensor: 入力テンソル
        target_layer: 可視化したい層
        target_class: 可視化したいクラス（Noneの場合は予測クラス）
        save_path: 保存先のパス（Noneの場合は表示のみ）
    """
    # Grad-CAMのインスタンスを作成
    grad_cam = GradCAM(model, target_layer)
    
    # CAMを生成
    cam = grad_cam.generate_cam(input_tensor, target_class)
    
    # 入力画像を取得
    input_image = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    
    # ヒートマップを入力画像のサイズにリサイズ
    import cv2
    cam_resized = cv2.resize(cam, (input_image.shape[1], input_image.shape[0]))
    
    # ヒートマップの色を調整
    heatmap = plt.cm.jet(cam_resized)[..., :3]  # RGBAからRGBに変換
    
    # ヒートマップと元画像を合成
    alpha = 0.5  # ヒートマップの透明度
    output = (1 - alpha) * input_image + alpha * heatmap
    
    # 可視化
    plt.figure(figsize=(15, 5))
    
    # 元の画像
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # ヒートマップ
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.axis('off')
    
    # 合成画像
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    import os
    from torchinfo import summary
    from torchviz import make_dot

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    mdl = build_model()
    print(mdl)
    summary(mdl, (batchSize, 3, cellSize, cellSize))
    
    x = torch.randn(batchSize, 3, cellSize, cellSize).to(DEVICE) # 適当な入力
    y = mdl(x) # その出力
    
    # 特徴マップの可視化
    visualize_feature_maps(mdl, x, 'features', 'feature_maps.png')
    
    # Grad-CAMの可視化
    target_layer = mdl.model_pre.features[-1]  # EfficientNetV2の最後の畳み込み層
    visualize_gradcam(mdl, x, target_layer, save_path='gradcam.png')
    
    img = make_dot(y, params = dict(mdl.named_parameters())) # 計算グラフの表示
    img.format = "png"
    img.render("_model_graph") # グラフを画像に保存
    os.remove("_model_graph") # 拡張子無しのファイルもできるので個別に削除