import os
from flask import Flask, render_template, request
from werkzeug.datastructures import FileStorage
import io, pkgutil
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
 
model = pickle.load(open("dipmodel.pkl","rb"))

#ファイルのアップデートを読み込む
file = request.files['csvfile']
#読み込んだデータを読み込み(ファイル形式はcsv)
uni_string = file.stream.read()     
#BytesIO はインメモリーのバイナリストリーム
df = pd.read_csv(io.BytesIO(uni_string), encoding='utf8')

ques_col = ['10時以降出社OK',
 '16時前退社OK',
 '1日7時間以下勤務OK',
 'Accessのスキルを活かす',
 'Excelのスキルを活かす',
 'PCスキル不要',
 'PowerPointのスキルを活かす',
 'Wordのスキルを活かす',
 'オフィスが禁煙・分煙',
 'シフト勤務',
 'フラグオプション選択',
 '交通費別途支給',
 '仕事の仕方',
 '休日休暇(土曜日)',
 '休日休暇(日曜日)',
 '休日休暇(月曜日)',
 '休日休暇(木曜日)',
 '休日休暇(水曜日)',
 '休日休暇(火曜日)',
 '休日休暇(祝日)',
 '休日休暇(金曜日)',
 '会社概要\u3000業界コード',
 '制服あり',
 '勤務先公開',
 '勤務地\u3000市区町村コード',
 '勤務地\u3000最寄駅1（分）',
 '勤務地\u3000最寄駅1（駅からの交通手段）',
 '勤務地\u3000最寄駅2（分）',
 '勤務地\u3000都道府県コード',
 '土日祝のみ勤務',
 '土日祝休み',
 '外資系企業',
 '大手企業',
 '大量募集',
 '学校・公的機関（官公庁）',
 '平日休みあり',
 '扶養控除内',
 '服装自由',
 '未経験OK',
 '正社員登用あり',
 '残業なし',
 '残業月20時間以上',
 '残業月20時間未満',
 '派遣スタッフ活躍中',
 '派遣形態',
 '短時間勤務OK(1日4h以内)',
 '社員食堂あり',
 '紹介予定派遣',
 '経験者優遇',
 '給与/交通費\u3000交通費',
 '給与/交通費\u3000給与上限',
 '給与/交通費\u3000給与下限',
 '職場の様子',
 '職種コード',
 '英語以外の語学力を活かす',
 '英語力を活かす',
 '英語力不要',
 '車通勤OK',
 '週2・3日OK',
 '週4日勤務',
 '駅から徒歩5分以内',
 '（派遣先）配属先部署\u3000人数',
 '（派遣先）配属先部署\u3000平均年齢',
 '（派遣先）配属先部署\u3000男女比\u3000女',
 '（派遣先）配属先部署\u3000男女比\u3000男']

@app.route('/')
def index():
    return render_template('index.html')
    


@app.route('/uploads', methods=['POST'])
def infer():
    #カラムの並べ替え
    ques_x = df.loc[:, ques_col]
    #欠損値処理（平均値で穴埋め）
    ques_x['給与/交通費\u3000給与上限'].fillna(ques_x['給与/交通費\u3000給与上限'].mean(), inplace=True)
    ques_x['（派遣先）配属先部署\u3000平均年齢'].fillna(ques_x['（派遣先）配属先部署\u3000平均年齢'].mean(), inplace=True)
    #欠損値処理（中央値で穴埋め）
    ques_x['勤務地\u3000最寄駅1（分）'].fillna(ques_x['勤務地\u3000最寄駅1（分）'].median(), inplace=True)
    ques_x['勤務地\u3000最寄駅2（分）'].fillna(ques_x['勤務地\u3000最寄駅2（分）'].median(), inplace=True)
    ques_x['（派遣先）配属先部署\u3000人数'].fillna(ques_x['（派遣先）配属先部署\u3000人数'].median(), inplace=True)
    #欠損値処理（最頻値で穴埋め）
    ques_x['勤務地\u3000最寄駅1（駅からの交通手段）'].fillna(ques_x['勤務地\u3000最寄駅1（駅からの交通手段）'].mode(), inplace=True)
    #欠損値処理（固定値で穴埋め）
    ques_x['勤務地\u3000最寄駅1（駅からの交通手段）'].fillna(1.0, inplace=True)
    ques_x['（派遣先）配属先部署\u3000男女比\u3000女'].fillna(5.0, inplace=True)
    ques_x['（派遣先）配属先部署\u3000男女比\u3000男'].fillna(5.0, inplace=True) 
    #モデル適応
    ques_y_ans = model.predict(ques_x)

    #答えのデータフレーム作成
    answ1 = pd.DataFrame(df.loc[:, 'お仕事No.'])
    answ2 = pd.DataFrame(ques_y_ans, columns=['応募数 合計'])
    answ = pd.concat([answ1,answ2], axis=1)

    resp = make_response(answ.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp
 
if __name__ == "__main__":
    app.run()