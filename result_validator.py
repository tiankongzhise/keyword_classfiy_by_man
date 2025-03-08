import pandas as pd
import os
from typing import Dict, List, Set
from tkinter import filedialog, messagebox
import tkinter as tk

def compare_excel_files(file1: str, file2: str) -> bool:
    """比较两个Excel文件的内容是否一致"""
    try:
        # 读取两个Excel文件
        df1_all = pd.read_excel(file1, sheet_name='全部去重结果')
        df2_all = pd.read_excel(file2, sheet_name='全部去重结果')
        
        # 比较全部去重结果sheet
        if not df1_all.equals(df2_all):
            print(f"'全部去重结果'sheet内容不一致")
            return False
            
        # 获取所有sheet名称
        with pd.ExcelFile(file1) as xls1, pd.ExcelFile(file2) as xls2:
            sheets1 = set(xls1.sheet_names)
            sheets2 = set(xls2.sheet_names)
            
        # 比较sheet名称是否一致
        if sheets1 != sheets2:
            print(f"sheet名称不一致：\n文件1: {sheets1}\n文件2: {sheets2}")
            return False
            
        # 比较每个分组sheet的内容
        for sheet in sheets1:
            if sheet == '全部去重结果':
                continue
                
            df1 = pd.read_excel(file1, sheet_name=sheet)
            df2 = pd.read_excel(file2, sheet_name=sheet)
            
            if not df1.equals(df2):
                print(f"'{sheet}'sheet内容不一致")
                return False
                
        return True
        
    except Exception as e:
        print(f"比较过程中出现错误：{str(e)}")
        return False

def main():
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    try:
        # 选择第一个文件（单线程版本结果）
        file1 = filedialog.askopenfilename(
            title='选择单线程版本结果文件',
            filetypes=[("Excel files", "*_classified.xlsx")]
        )
        if not file1:
            return
            
        # 选择第二个文件（多线程版本结果）
        file2 = filedialog.askopenfilename(
            title='选择多线程版本结果文件',
            filetypes=[("Excel files", "*_classified_mp.xlsx")]
        )
        if not file2:
            return
            
        # 比较文件内容
        is_identical = compare_excel_files(file1, file2)
        
        # 显示比较结果
        if is_identical:
            messagebox.showinfo('比较结果', '两个文件的内容完全一致！')
        else:
            messagebox.showerror('比较结果', '两个文件的内容存在差异！')
            
    except Exception as e:
        messagebox.showerror('错误', f'程序执行过程中出现错误：\n{str(e)}')
    finally:
        root.destroy()

if __name__ == '__main__':
    main()