import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from typing import List, Dict
import os
import time

class KeywordClassifier:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('关键词分类工具')
        self.window.geometry('600x400')
        
        # 创建界面组件
        self.create_widgets()
        
    def create_widgets(self):
        # 词根输入框
        tk.Label(self.window, text='请输入词根（用逗号分隔）：').pack(pady=5)
        self.root_entry = tk.Text(self.window, height=3)
        self.root_entry.pack(pady=5, padx=10, fill=tk.X)
        
        # 文件选择按钮
        frame = ttk.Frame(self.window)
        frame.pack(pady=10, fill=tk.X, padx=10)
        
        ttk.Label(frame, text='词根文件：').pack(side=tk.LEFT)
        self.root_file_path = tk.StringVar()
        ttk.Entry(frame, textvariable=self.root_file_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frame, text='选择', command=self.select_root_file).pack(side=tk.RIGHT)
        
        frame2 = ttk.Frame(self.window)
        frame2.pack(pady=10, fill=tk.X, padx=10)
        
        ttk.Label(frame2, text='目标文件：').pack(side=tk.LEFT)
        self.target_file_path = tk.StringVar()
        ttk.Entry(frame2, textvariable=self.target_file_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frame2, text='选择', command=self.select_target_file).pack(side=tk.RIGHT)
        
        # 忽略大小写选项
        self.ignore_case = tk.BooleanVar()
        ttk.Checkbutton(self.window, text='忽略大小写', variable=self.ignore_case).pack(pady=5)
        
        # 处理按钮
        ttk.Button(self.window, text='开始处理', command=self.process_keywords).pack(pady=20)
        
    def select_root_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.root_file_path.set(file_path)
            
    def select_target_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        if file_path:
            self.target_file_path.set(file_path)
            
    def process_keywords(self):
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 获取词根
            roots_text = self.root_entry.get('1.0', tk.END).strip()
            manual_roots = [r.strip() for r in roots_text.split(',') if r.strip()]
            
            # 读取词根文件
            if self.root_file_path.get():
                root_df = pd.read_csv(self.root_file_path.get())
                file_roots = root_df.iloc[:, 0].tolist()
                roots = list(set(manual_roots + file_roots))
            else:
                roots = manual_roots
                
            if not roots:
                messagebox.showerror('错误', '请输入词根或选择词根文件')
                return
                
            # 读取目标文件
            target_file = self.target_file_path.get()
            if not target_file:
                messagebox.showerror('错误', '请选择要处理的目标文件')
                return
                
            if target_file.endswith('.xlsx'):
                df = pd.read_excel(target_file)
            else:
                df = pd.read_csv(target_file)
                
            # 清洗关键词，去除花括号
            df.iloc[:, 0] = df.iloc[:, 0].str.replace('{', '').str.replace('}', '')
                
            # 分类关键词
            keyword_groups: Dict[str, List[str]] = {}
            all_keywords = set()
            
            for root in roots:
                # 根据是否忽略大小写设置匹配条件
                if self.ignore_case.get():
                    keywords = df[df.iloc[:, 0].str.lower().str.contains(root.lower(), na=False)].iloc[:, 0].unique().tolist()
                else:
                    keywords = df[df.iloc[:, 0].str.contains(root, na=False)].iloc[:, 0].unique().tolist()
                keyword_groups[root] = keywords
                all_keywords.update(keywords)
                
            # 创建结果Excel文件
            output_file = os.path.splitext(target_file)[0] + '_classified.xlsx'
            with pd.ExcelWriter(output_file) as writer:
                # 第一个sheet：全部去重结果
                all_data = []
                used_keywords = set()
                
                for root, keywords in keyword_groups.items():
                    for kw in keywords:
                        status = '保留' if kw not in used_keywords else f'重复（被{next(r for r, kws in keyword_groups.items() if kw in kws and r != root)}组删除）'
                        if kw not in used_keywords:
                            used_keywords.add(kw)
                        all_data.append({'关键词': kw, '词根': root, '状态': status})
                        
                pd.DataFrame(all_data).to_excel(writer, sheet_name='全部去重结果', index=False)
                
                # 每个词根一个sheet
                for root, keywords in keyword_groups.items():
                    data = []
                    for kw in keywords:
                        data.append({'关键词': kw, '词根': root})
                    pd.DataFrame(data).to_excel(writer, sheet_name=root, index=False)
                    
            messagebox.showinfo('完成', f'处理完成！总耗时：{time.time() - start_time:.2f}秒\n结果已保存至：\n{output_file}')
            
        except Exception as e:
            messagebox.showerror('错误', f'处理过程中出现错误：\n{str(e)}')
            
    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    app = KeywordClassifier()
    app.run()
