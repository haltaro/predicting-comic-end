# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
from time import sleep

from matplotlib.font_manager import FontProperties
font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)

sns.set(style='ticks')


def search_magazine(key='JUMPrgl', n_pages=25):
    """
    「ユニークID」「雑誌巻号ID」あるいは「雑誌コード」にkey含む雑誌を，
    n_pages分取得する関数です．
    """
    
    url = 'https://mediaarts-db.bunka.go.jp/mg/api/v1/results_magazines?id=' + \
        key + '&page='
    magazines = []
    
    for i in range(1, n_pages):
        response = urllib.request.urlopen(url + str(i))
        content = json.loads(response.read().decode('utf8'))
        magazines.extend(content['results'])
    return magazines


def extract_data(content):
    """
    contentに含まれる目次情報を取得する関数です．
    - year: 発行年
    - no: 号数
    - title: 作品名
    - author: 著者
    - color: カラーか否か
    - pages: 掲載ページ数
    - start_page: 作品のスタートページ
    - best: 巻頭から数えた掲載順
    - worst: 巻末から数えた掲載順
    """
    
    # マンガ作品のみ抽出します．
    comics = [comic for comic in content['contents'] 
             if comic['category']=='マンガ作品'] 
    data = []
    year = int(content['basics']['date_indication'][:4])
    
    # 号数が記載されていない場合があるので，例外処理が必要です．
    try:
        no = int(content['basics']['number_indication'])
    except ValueError:
        no = content['basics']['number_indication']
    
    for comic in comics:
        title= comic['work']
        if not title:
            continue
            
        # ページ数が記載されていない作品があるので，例外処理が必要です．
        # 特に理由はないですが，無記載の作品は10ページとして処理を進めます．
        try:
            pages = int(comic['work_pages'])
        except ValueError:
            pages = 10

        # 「いぬまるだしっ」等，1週に複数話掲載されている作品に対応するため
        # data中にすでにtitleが含まれる場合は，新規datumとして登録せずに，
        # 既存のdatumのページ数のみ加算します．
        if len(data) > 0 and title in [datum['title'] for datum in data]:
            data[[datum['title'] for datum in 
                  data].index(title)]['pages'] += pages
        else:
            data.append({
                'year': year,
                'no': no,
                'title': comic['work'],
                'author': comic['author'],
                'subtitle': comic['subtitle'],
                'color': int('カラー' in comic['note']),
                'pages': int(comic['work_pages']),
                'start_pages': int(comic['start_page'])
            })

    # 企画物のミニマンガを除外するため，合計5ページ以下のdatumはリストから除外します．
    filterd_data = [datum for datum in data if datum['pages'] > 5]
    for n, datum in enumerate(filterd_data):
        datum['best'] = n + 1
        datum['worst'] = len(filterd_data) - n
        
    return filterd_data


def save_data(magazines, offset=0, file_name='data/wj-api.json'):
    """
    magazinesに含まれる全てのmagazineについて，先頭からoffset以降の巻号の
    目次情報を取得し，file_nameに保存する関数です．
    """
    
    url = 'https://mediaarts-db.bunka.go.jp/mg/api/v1/magazine?id='
    
    #　ファイル先頭行
    if offset == 0:
        with open(file_name, 'w') as f:
            f.write('[\n')
        
    with open(file_name, 'a') as f:
        
        # magazines中のmagazine毎にWeb APIを叩きます．
        for m, magazine in enumerate(magazines[offset:]):
            response = urllib.request.urlopen(url + str(magazine['id']),
                                              timeout=30)
            content = json.loads(response.read().decode('utf8'))
            
            # 前記の関数extract_data()で，必要な情報を抽出します．
            comics = extract_data(content)
            print('{0:4d}/{1}: Extracted data from {2}'.\
                  format(m + offset, len(magazines), url + str(magazine['id'])))
            
            # comics中の各comicについて，file_nameに情報を保存します．
            for n, comic in enumerate(comics):
                
                # ファイル先頭以外の，magazineの最初のcomicの場合は，
                # まず',\n'を追記．
                if m + offset > 0 and n == 0:
                    f.write(',\n')
                
                json.dump(comic, f, ensure_ascii=False)
                
                # 最後のcomic以外は',\n'を追記．
                if not n == len(comics) - 1:
                    f.write(',\n')
            print('{0:9}: Saved data to {1}'.format(' ', file_name))
            
            # サーバへの負荷を抑えるため，必ず一時停止します．
            sleep(3)
            
    # ファイル最終行
    with open(file_name, 'a') as f:
        f.write(']')


class ComicAnalyzer():
    """漫画雑誌の目次情報を読みだして，管理するクラスです．"""

    def __init__(self, data_path='data/wj-api.json', min_week=7,
                 short_week=10):
        """
        初期化時に，data_pathにある.jsonファイルから目次情報を抽出します．
        - self.data: 全目次情報を保持するリスト型
        - self.all_titles: 全作品名情報を保持するリスト型
        - self.serialized_titles: min_week以上連載した全作品名を保持するリスト型
        - self.last_year: 最新の目次情報の年を保持する数値型
        - self.last_no: 最新の目次情報の号数を保持する数値型
        - self.end_titles: self.serialized_titlesのうち，self.last_yearおよび
                           self.last_noまでに終了した全作品名を保持するリスト型
        - self.short_end_titles: self.end_titlesのうち，short_week週以内に
                                 連載が終了した作品名を保持するリスト型
        - self.long_end_titles: self.end_titlesのうち，short_week+1週以上に
                                連載が継続した作品名を保持するリスト型
        """

        self.data = self.read_data(data_path)
        self.all_titles = self.collect_all_titles()
        self.serialized_titles = self.drop_short_titles(
            self.all_titles, min_week)
        self.last_year = self.find_last_year(self.serialized_titles[-100:])
        self.last_no = self.find_last_no(self.serialized_titles[-100:],
                                         self.last_year)
        self.end_titles = self.drop_continued_titles(
            self.serialized_titles, self.last_year, self.last_no)
        self.short_end_titles = self.drop_long_titles(
            self.end_titles, short_week)
        self.long_end_titles = self.drop_short_titles(
            self.end_titles, short_week + 1)

    def read_data(self, data_path):
        """ data_pathにあるjsonファイルを読み出して，
            全ての目次情報をまとめたリストを返します． """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def collect_all_titles(self):
        """ self.dataから全ての作品名を抽出したリストを返します． """
        titles = []
        for comic in self.data:
            if comic['title'] not in titles:
                titles.append(comic['title'])
        return titles

    def extract_item(self, title='ONE PIECE', item='worst'):
        """ self.dataからtitleのitemをすべて抽出したリストを返します． """
        return [comic[item] for comic in self.data if comic['title'] == title]

    def drop_short_titles(self, titles, min_week):
        """ titlesのうち，min_week週以上連載した作品名のリストを返します． """
        return [title for title in titles
                if len(self.extract_item(title)) >= min_week]

    def drop_long_titles(self, titles, max_week):
        """ titlesのうち，max_week週以内で終了した作品名のリストを返します． """
        return [title for title in titles
                if len(self.extract_item(title)) <= max_week]

    def find_last_year(self, titles):
        """ titlesが掲載された雑誌のうち，最新の年を返します． """
        return max([self.extract_item(title, 'year')[-1]
                   for title in titles])

    def find_last_no(self, titles, year):
        """ titlesが掲載されたyear年の雑誌のうち，最新の号数を返します． """
        return max([self.extract_item(title, 'no')[-1]
                   for title in titles
                   if self.extract_item(title, 'year')[-1] == year])

    def drop_continued_titles(self, titles, year, no):
        """ titlesのうち，year年のno号までに連載が終了した作品名のリストを返します． """
        end_titles = []
        for title in titles:
            last_year = self.extract_item(title, 'year')[-1]
            if last_year < year:
                end_titles.append(title)
            elif last_year == year:
                if self.extract_item(title, 'no')[-1] < no:
                    end_titles.append(title)
        return end_titles

    def search_title(self, key, titles):
        """ titlesのうち，keyを含む作品名のリストを返します． """
        return [title for title in titles if key in title]


class ComicNet():
    """ マンガ作品が短命か否かを識別する多層パーセプトロンを管理するクラスです．  
    :param thresh_week：短命作品とそれ以外を分けるしきい値．
    :param n_x：多層パーセプトロンに入力する掲載週の数．入力層のノード数．
    """
    def __init__(self, thresh_week=20, n_x=7):
        self.n_x = n_x
        self.thresh_week = thresh_week        
    
    def get_x(self, analyzer, title):
        """指定された作品の指定週までの正規化掲載順を取得する関数です．"""
        worsts = np.array(analyzer.extract_item(title)[:self.n_x])
        bests = np.array(analyzer.extract_item(title, 'best')[:self.n_x])
        bests_normalized = bests / (worsts + bests - 1)
        color = sum(analyzer.extract_item(title, 'color')[:self.n_x]
                    ) /self.n_x
        return np.append(bests_normalized, color)

    def get_y(self, analyzer, title, thresh_week):
        """指定された作品が，短命作品か否かを取得する関数です．"""
        return int(len(analyzer.extract_item(title)) <=  thresh_week)

    def get_xs_ys(self, analyzer, titles, thresh_week):
        """指定された作品群の特徴量とラベルとタイトルを返す関数です．
        　　y==0とy==1のデータ数を揃えて返します．
        """
        xs = np.array([self.get_x(analyzer, title) for title in titles])
        ys = np.array([[self.get_y(analyzer, title, thresh_week)] 
                       for title in titles])
        
        # ys==0とys==1のデータ数を揃えます．
        idx_ps = np.where(ys.reshape((-1)) == 1)[0]
        idx_ng = np.where(ys.reshape((-1)) == 0)[0]
        len_data = min(len(idx_ps), len(idx_ng))
        x_ps = xs[idx_ps[-len_data:]]
        x_ng = xs[idx_ng[-len_data:]]
        y_ps = ys[idx_ps[-len_data:]]
        y_ng = ys[idx_ng[-len_data:]]
        t_ps = [titles[ii] for ii in idx_ps[-len_data:]]
        t_ng = [titles[ii] for ii in idx_ng[-len_data:]]
        
        return x_ps, x_ng, y_ps, y_ng, t_ps, t_ng
        
    def augment_x(self, x, n_aug):
        """指定された数のxデータを人為的に生成する関数です．"""
        if n_aug:
            x_pair = np.array(
                [[x[idx] for idx in 
                  np.random.choice(range(len(x)), 2, replace=False)]
                 for _ in range(n_aug)])
            weights = np.random.rand(n_aug, 1, self.n_x + 1)
            weights = np.concatenate((weights, 1 - weights), axis=1)
            x_aug = (x_pair * weights).sum(axis=1)
            
            return np.concatenate((x, x_aug), axis=0)
        else:
            return x
        
    def augment_y(self, y, n_aug):
        """指定された数のyデータを人為的に生成する関数です．"""
        if n_aug:
            y_aug = np.ones((n_aug, 1)) if y[0, 0] \
                else np.zeros((n_aug, 1))
            return np.concatenate((y, y_aug), axis=0)
        else:
            return y
        
    def configure_dataset(self, analyzer, n_drop=0, n_aug=0):
        """データセットを設定する関数です．
        :param analyzer: ComicAnalyzerクラスのインスタンス
        :param n_drop: trainingデータから除外する古いデータの数
        :param n_aug: trainingデータに追加するaugmentedデータの数
        """
        x_ps, x_ng, y_ps, y_ng, t_ps, t_ng = self.get_xs_ys(
            analyzer, analyzer.end_titles, self.thresh_week)
        self.x_test = np.concatenate((x_ps[-50:], x_ng[-50:]), axis=0)
        self.y_test = np.concatenate((y_ps[-50:], y_ng[-50:]), axis=0)
        self.titles_test = t_ps[-50:] + t_ng[-50:]
        self.x_val = np.concatenate((x_ps[-100 : -50], 
                                     x_ng[-100 : -50]), axis=0)
        self.y_val = np.concatenate((y_ps[-100 : -50], 
                                     y_ng[-100 : -50]), axis=0)
        self.x_tra = np.concatenate(
            (self.augment_x(x_ps[n_drop//2 : -100], n_aug//2), 
             self.augment_x(x_ng[n_drop//2 : -100], n_aug//2)), axis=0)
        self.y_tra = np.concatenate(
            (self.augment_y(y_ps[n_drop//2 : -100], n_aug//2), 
             self.augment_y(y_ng[n_drop//2 : -100], n_aug//2)), axis=0)
    
    def build_graph(self, r=0.001, n_h=7, stddev=0.01):
        """多層パーセプトロンを構築する関数です．
        :param r: 学習率
        :param n_h: 隠れ層のノード数
        :param stddev: 変数の初期分布の標準偏差
        """
        tf.reset_default_graph()
        
        # 入力層およびターゲット
        n_y = self.y_test.shape[1]
        self.x = tf.placeholder(tf.float32, [None, self.n_x + 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, n_y], name='y')
        
        # 隠れ層（１層目）
        self.w_h_1 = tf.Variable(
            tf.truncated_normal((self.n_x + 1, n_h), stddev=stddev))
        self.b_h_1 = tf.Variable(tf.zeros(n_h))
        self.logits = tf.add(tf.matmul(self.x, self.w_h_1), self.b_h_1)
        self.logits = tf.nn.relu(self.logits)
        
        # 隠れ層（２層目）
        self.w_h_2 = tf.Variable(
            tf.truncated_normal((n_h, n_h), stddev=stddev))
        self.b_h_2 = tf.Variable(tf.zeros(n_h))
        self.logits = tf.add(tf.matmul(self.logits, self.w_h_2), self.b_h_2)
        self.logits = tf.nn.relu(self.logits)
        
        # 出力層
        self.w_y = tf.Variable(
            tf.truncated_normal((n_h, n_y), stddev=stddev))
        self.b_y = tf.Variable(tf.zeros(n_y))
        self.logits = tf.add(tf.matmul(self.logits, self.w_y), self.b_y)
        tf.summary.histogram('logits', self.logits)
        
        # 損失関数
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.y))
        tf.summary.scalar('loss', self.loss)
        
        # 最適化
        self.optimizer = tf.train.AdamOptimizer(r).minimize(self.loss)
        self.output = tf.nn.sigmoid(self.logits, name='output')
        correct_prediction = tf.equal(self.y, tf.round(self.output))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
            name='acc')
        tf.summary.histogram('output', self.output)
        tf.summary.scalar('acc', self.acc)
        
        self.merged = tf.summary.merge_all()
            
        
    def train(self, epoch=2000, print_loss=False, save_log=False, 
              log_dir='./logs/1', log_name='', save_model=False,
              model_name='prediction_model'):
        """多層パーセプトロンを学習させ，ログや学習済みモデルを保存する関数です．
        :param epoch: エポック数
        :pram print_loss: 損失関数の履歴を出力するか否か
        :param save_log: ログを保存するか否か
        :param log_dir: ログの保存ディレクトリ
        :param log_name: ログの保存名
        :param save_model: 学習済みモデルを保存するか否か
        :param model_name: 学習済みモデルの保存名
        """
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) # 変数の初期化
            
            # ログ保存用の設定
            log_tra = log_dir + '/tra/' + log_name 
            writer_tra = tf.summary.FileWriter(log_tra)
            log_val = log_dir + '/val/' + log_name
            writer_val = tf.summary.FileWriter(log_val)        

            for e in range(epoch):
                feed_dict = {self.x: self.x_tra, self.y: self.y_tra}
                _, loss_tra, acc_tra, mer_tra = sess.run(
                        (self.optimizer, self.loss, self.acc, self.merged), 
                        feed_dict=feed_dict)
                
                # validation
                feed_dict = {self.x: self.x_val, self.y: self.y_val}
                loss_val, acc_val, mer_val = sess.run(
                    (self.loss, self.acc, self.merged),
                    feed_dict=feed_dict)
                
                # ログの保存
                if save_log:
                    writer_tra.add_summary(mer_tra, e)
                    writer_val.add_summary(mer_val, e)
                
                # 損失関数の出力
                if print_loss and e % 500 == 0:
                    print('# epoch {}: loss_tra = {}, loss_val = {}'.
                          format(e, str(loss_tra), str(loss_val)))
            
            # モデルの保存
            if save_model:
                saver = tf.train.Saver()
                _ = saver.save(sess, './models/' + model_name)
            
    def test(self, model_name='prediction_model'):
        """指定されたモデルを読み込み，テストする関数です．
        :param model_name: 読み込むモデルの名前
        """
        tf.reset_default_graph()
        loaded_graph = tf.Graph()
        
        with tf.Session(graph=loaded_graph) as sess:
            
            # モデルの読み込み
            loader = tf.train.import_meta_graph(
                './models/{}.meta'.format(model_name))
            loader.restore(sess, './models/' + model_name)
            
            x_loaded = loaded_graph.get_tensor_by_name('x:0')
            y_loaded = loaded_graph.get_tensor_by_name('y:0')
            
            loss_loaded = loaded_graph.get_tensor_by_name('loss:0')
            acc_loaded = loaded_graph.get_tensor_by_name('acc:0')
            output_loaded = loaded_graph.get_tensor_by_name('output:0')
        
            # test
            feed_dict = {x_loaded: self.x_test, y_loaded: self.y_test}
            loss_test, acc_test, output_test = sess.run(
                (loss_loaded, acc_loaded, output_loaded), feed_dict=feed_dict)
            return acc_test, output_test
