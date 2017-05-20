import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='ticks')

import matplotlib
from matplotlib.font_manager import FontProperties
font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)


class ComicAnalyzer():
    """漫画雑誌の目次情報を読みだして，管理するクラスです．"""

    def __init__(self, data_path='data/wj-api.json', min_week=7, short_week=10):
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
        self.serialized_titles = self.drop_short_titles(self.all_titles, min_week)
        self.last_year = self.find_last_year(self.serialized_titles[-100:])
        self.last_no = self.find_last_no(self.serialized_titles[-100:], self.last_year)
        self.end_titles = self.drop_continued_titles(
            self.serialized_titles, self.last_year, self.last_no)
        self.short_end_titles = self.drop_long_titles(
            self.end_titles, short_week)
        self.long_end_titles = self.drop_short_titles(
            self.end_titles, short_week + 1)

    def read_data(self, data_path):
        """ data_pathにあるjsonファイルを読み出して，全ての目次情報をまとめたリストを返します． """
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
