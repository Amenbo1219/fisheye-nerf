import json
import math
ROTATION = True
def calc_rm(yaw,pitch,roll):
    # 各軸の回転行列を計算
    Rz = [
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ]
    
    Ry = [
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ]
    
    Rx = [
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ]
    
    # 行列の積を計算（Rz * Ry * Rx）
    R = matrix_multiply(Rz, matrix_multiply(Ry, Rx))
    R = [item for sublist in R for item in sublist]
    R = [round(i,2) for i in R]
    return R
def matrix_multiply(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def mk_poses(read_json,out_txt):
    # reconstruction.jsonファイルを読み込む
    with open(read_json, 'r') as f:
        data = json.load(f)

    data = data[0]
    # poses.txtファイルを開く
    with open(out_txt, 'w') as f:
        cnt =0
        o_x = 0
        o_y = 0
        o_z = 0
        #print(data[0]['shots'])
        for shot_id, shot in data['shots'].items():
            # ファイル名から拡張子を除去
            name = shot_id.split('.')[0]
            
            # カメラの位置(x, y, z)を取得
            x, y, z = shot['translation']
            n = 1
            x, y, z = round(x/n,2) , round(y/n,2) , round(z/n,2)
            # 単位行列(0, -1, 0, 1, 0, 0)を使用して回転行列を作成
            r = [1.0,0.0,0.0,0.0,1.0,0.0, 0.0, 0.0,  1.0]
            r_x,r_y,r_z = shot['rotation']
            r = calc_rm(r_x,r_y,r_z)

            # if ROTATION==True:
            #     if cnt==0:
            #         o_x,o_y,o_z = shot['rotation']
            #         r_x,r_y,r_z = 0,0,0     
            #     else :
            #         r_x,r_y,r_z = shot['rotation']
            #         r_x -= o_x
            #         r_y -= o_y
            #         r_z -= o_z
            #     r = calc_rm(r_x,r_y,r_z)

            cnt += 1
            # poses.txtファイルに書き込む
            f.write(f'{name} {r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]} {r[7]} {r[8]} {x} {y} {z}\n')
if __name__ == '__main__':
    import argparse

    # 引数パーサを作成
    parser = argparse.ArgumentParser()

    # 引数を定義
    parser.add_argument("-i","--input", default="reconstruction.json",help="入力")
    parser.add_argument("-o","--output", default="poses.txt",help="出力")

    # 引数を解析
    args = parser.parse_args()
    mk_poses(args.input,args.output)