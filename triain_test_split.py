import random
import os
root = './'
# root = 'workspace/nerf-360-orign/data/Blender-Classroomv2'
# ファイルからデータを読み込む
with open(os.path.join(root,'poses.txt'), 'r') as file:
    data = file.readlines()

# データをランダムにシャッフル
random.shuffle(data)

# 分割ポイントを計算
split_point = int(len(data) * 0.8)

# データを分割
train_data = data[:split_point]
test_data = data[split_point:]

# 分割したデータを新しいファイルに書き込む
with open(os.path.join(root,'train_data.txt'), 'w') as file:
    file.writelines(train_data)

with open(os.path.join(root,'test_data.txt'), 'w') as file:
    file.writelines(test_data)