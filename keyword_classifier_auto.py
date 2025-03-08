import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from typing import List, Dict, Set
import os
import time
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
from collections import defaultdict

class KeywordClassifierAuto:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('关键词分类工具（自动分组版）')
        self.window.geometry('600x400')
        
        # 创建界面组件
        self.create_widgets()
        
    def create_widgets(self):
        # 文件选择按钮
        frame = ttk.Frame(self.window)
        frame.pack(pady=10, fill=tk.X, padx=10)
        
        ttk.Label(frame, text='目标文件：').pack(side=tk.LEFT)
        self.target_file_path = tk.StringVar()
        ttk.Entry(frame, textvariable=self.target_file_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frame, text='选择', command=self.select_target_file).pack(side=tk.RIGHT)
        
        # 参数设置
        frame2 = ttk.Frame(self.window)
        frame2.pack(pady=10, fill=tk.X, padx=10)
        
        ttk.Label(frame2, text='最小相似度阈值：').pack(side=tk.LEFT)
        self.similarity_threshold = tk.DoubleVar(value=0.6)
        ttk.Entry(frame2, textvariable=self.similarity_threshold, width=10).pack(side=tk.LEFT)
        
        ttk.Label(frame2, text='最小组大小：').pack(side=tk.LEFT, padx=(20, 0))
        self.min_group_size = tk.IntVar(value=3)
        ttk.Entry(frame2, textvariable=self.min_group_size, width=10).pack(side=tk.LEFT)
        
        # 处理按钮
        ttk.Button(self.window, text='开始处理', command=self.process_keywords).pack(pady=20)
        
    def select_target_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        if file_path:
            self.target_file_path.set(file_path)
    
    def extract_features(self, keywords: List[str]) -> List[List[str]]:
        """使用jieba分词提取关键词特征"""
        try:
            features = []
            for keyword in keywords:
                # 对每个关键词进行分词，并过滤空字符
                words = [w for w in jieba.cut(keyword) if w.strip()]
                if not words:  # 如果分词结果为空，使用原始关键词
                    words = [keyword]
                features.append(words)
            return features
        except Exception as e:
            print(f"分词错误: {str(e)}")
            return [[k] for k in keywords]  # 出错时返回原始关键词作为特征
    
    def train_word2vec(self, features: List[List[str]]) -> Word2Vec:
        """训练词向量模型"""
        try:
            # 减小向量维度，优化训练参数
            model = Word2Vec(sentences=features,
                          vector_size=50,  # 减小向量维度
                          window=3,        # 减小窗口大小
                          min_count=1,
                          workers=4,
                          epochs=5)         # 限制训练轮数
            return model
        except Exception as e:
            print(f"模型训练错误: {str(e)}")
            # 返回一个最小配置的模型
            return Word2Vec(sentences=[[""]], vector_size=50, min_count=1)
    
    def get_keyword_vector(self, keyword_words: List[str], model: Word2Vec) -> np.ndarray:
        """计算关键词的向量表示（使用词向量的平均值）"""
        try:
            vectors = []
            for word in keyword_words:
                if word in model.wv:
                    vector = model.wv[word]
                    if not np.any(np.isnan(vector)):
                        vectors.append(vector)
            
            if vectors:
                # 确保数据类型一致性
                vectors = np.array(vectors, dtype=np.float64)
                return np.mean(vectors, axis=0)
            return np.zeros(model.vector_size, dtype=np.float64)
        except Exception as e:
            print(f"向量计算错误: {str(e)}")
            return np.zeros(model.vector_size, dtype=np.float64)
    
    def cluster_keywords(self, keyword_vectors: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """使用简单的距离阈值聚类"""
        try:
            # 确保输入数据为二维数组
            if len(keyword_vectors.shape) == 1:
                keyword_vectors = keyword_vectors.reshape(1, -1)
            
            # 将NumPy数组转换为Python列表
            vectors = keyword_vectors.tolist()
            n = len(vectors)
            
            # 初始化标签，-1表示未分配
            labels = [-1] * n
            current_label = 0
            
            # 对每个未分配的点进行聚类
            for i in range(n):
                if labels[i] != -1:
                    continue
                    
                # 找到当前点的邻居
                neighbors = []
                for j in range(n):
                    if i == j:
                        continue
                        
                    # 计算余弦相似度
                    dot_product = sum(a * b for a, b in zip(vectors[i], vectors[j]))
                    norm_i = sum(x * x for x in vectors[i]) ** 0.5
                    norm_j = sum(x * x for x in vectors[j]) ** 0.5
                    
                    if norm_i == 0 or norm_j == 0:
                        similarity = 0
                    else:
                        similarity = dot_product / (norm_i * norm_j)
                    
                    # 如果相似度大于阈值，加入邻居集合
                    if similarity >= (1 - eps):
                        neighbors.append(j)
                
                # 如果邻居数量达到最小要求，形成一个新的簇
                if len(neighbors) + 1 >= min_samples:
                    labels[i] = current_label
                    for j in neighbors:
                        if labels[j] == -1:
                            labels[j] = current_label
                    current_label += 1
            
            return np.array(labels)
        except Exception as e:
            print(f"聚类错误: {str(e)}")
            return np.zeros(len(keyword_vectors), dtype=np.int32)
    
    def find_group_keywords(self, group_keywords: List[str]) -> str:
        """从一组关键词中找出最能代表该组的词根"""
        # 获取最短的关键词作为词根
        return min(group_keywords, key=len)
    
    def process_keywords(self):
        try:
            # 设置递归限制
            import sys
            sys.setrecursionlimit(1000)  # 设置合理的递归深度限制
            
            # 获取目标文件
            target_file = self.target_file_path.get()
            if not target_file:
                messagebox.showerror('错误', '请选择要处理的目标文件')
                return
            
            # 读取文件
            try:
                if target_file.endswith('.xlsx'):
                    df = pd.read_excel(target_file)
                else:
                    df = pd.read_csv(target_file)
            except Exception as e:
                messagebox.showerror('错误', f'读取文件失败: {str(e)}')
                return
            
            # 数据验证
            if df.empty:
                messagebox.showerror('错误', '文件中没有数据')
                return
            
            if len(df.columns) == 0:
                messagebox.showerror('错误', '文件格式不正确')
                return
            
            # 清洗关键词
            df.iloc[:, 0] = df.iloc[:, 0].str.replace('{', '').str.replace('}', '')
            keywords = df.iloc[:, 0].dropna().unique().tolist()
            
            if not keywords:
                messagebox.showerror('错误', '没有找到有效的关键词')
                return
            
            # 特征提取
            features = self.extract_features(keywords)
            
            # 训练词向量模型
            model = self.train_word2vec(features)
            
            # 获取关键词向量
            keyword_vectors = []
            for keyword_words in features:
                vector = self.get_keyword_vector(keyword_words, model)
                if not np.any(np.isnan(vector)):
                    keyword_vectors.append(vector)
                
            if not keyword_vectors:
                messagebox.showerror('错误', '无法生成有效的关键词向量')
                return
            
            # 转换为numpy数组
            keyword_vectors = np.array(keyword_vectors)
            
            # 聚类
            eps = self.similarity_threshold.get()
            min_samples = self.min_group_size.get()
            
            try:
                labels = self.cluster_keywords(keyword_vectors, eps, min_samples)
            except Exception as e:
                messagebox.showerror('错误', f'聚类过程出错: {str(e)}')
                return
            
            # 处理结果
            groups = defaultdict(list)
            for keyword, label in zip(keywords, labels):
                if label != -1:  # 忽略噪声点
                    groups[f'组_{label}'].append(keyword)
            
            # 保存结果
            output_file = os.path.splitext(target_file)[0] + '_auto_classified.xlsx'
            try:
                with pd.ExcelWriter(output_file) as writer:
                    # 保存分组结果
                    all_data = []
                    for group_name, group_keywords in groups.items():
                        for keyword in group_keywords:
                            all_data.append({'关键词': keyword, '分组': group_name})
                    
                    if all_data:
                        pd.DataFrame(all_data).to_excel(writer, sheet_name='自动分类结果', index=False)
                    else:
                        pd.DataFrame(columns=['关键词', '分组']).to_excel(writer, sheet_name='自动分类结果', index=False)
                
                messagebox.showinfo('完成', f'处理完成，结果已保存至：\n{output_file}')
            except Exception as e:
                messagebox.showerror('错误', f'保存结果时出错: {str(e)}')
                
        except Exception as e:
            messagebox.showerror('错误', f'处理过程出错: {str(e)}')
            print(f"处理错误: {str(e)}")
    
    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    app = KeywordClassifierAuto()
    app.run()
