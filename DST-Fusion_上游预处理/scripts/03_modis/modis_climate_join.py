import pandas as pd
import os  # 这个工具用来处理文件路径，防止写错

# ================= 配置区 =================
# 1. 设置你的文件夹路径 (前面的 r 不能少)
# 植被数据的文件夹
dir_veg = r"E:\data\modis\Final_Results_Processed"
# 气候数据的文件夹
dir_cli = r"E:\data\气候指数\吉林"

# 2. 准备一个空箱子，用来装每一年处理好的数据
all_years_data = []
# ==========================================

print("=== 全自动合并流水线启动 ===")

# 3. 开始循环：从 2001 年跑到 2019 年
for year in range(2001, 2020):
    print(f"\n正在处理 {year} 年的数据...")

    # --- 第一步：动态拼凑文件名 ---
    # f"..." 的意思是：把大括号 {year} 里的数字自动替换掉
    file_veg = os.path.join(dir_veg, f"Jilin_{year}_Stats_Full.csv")
    file_cli = os.path.join(dir_cli, f"吉林_气候数据_{year}.csv")

    # 检查一下文件在不在，万一哪一年缺了文件，就跳过，别报错
    if not os.path.exists(file_veg) or not os.path.exists(file_cli):
        print(f"  警告：找不到 {year} 年的文件！跳过！")
        continue

    # --- 第二步：读取数据 ---
    try:
        # 之前你用 gbk 成功了，我们就继续用 gbk
        df_veg = pd.read_csv(file_veg, encoding='gbk')
        df_cli = pd.read_csv(file_cli, encoding='gbk')
    except:
        # 如果报错，试试 utf-8 (双保险)
        df_veg = pd.read_csv(file_veg, encoding='utf-8')
        df_cli = pd.read_csv(file_cli, encoding='utf-8')

    # --- 第三步：统一列名和日期 ---
    # 1. 统一县名：把 '县名' 改成 'County'
    if '县名' in df_veg.columns:
        df_veg = df_veg.rename(columns={'县名': 'County'})

    # 2. 统一日期：都生成一个叫 'Date' 的新列
    # 处理植被表：A2001106 -> 标准日期
    df_veg['Date'] = pd.to_datetime(df_veg['日期'].astype(str).str.replace('A', ''), format='%Y%j')
    # 处理气候表：time -> 标准日期
    df_cli['Date'] = pd.to_datetime(df_cli['time'])

    # --- 第四步：合并 (拼图) ---
    merged = pd.merge(df_veg, df_cli, on=['Date', 'County'], how='inner')

    # --- 第五步：【减肥】删除多余列 ---
    # 把原始的、乱七八糟的日期列都删掉，只留 'Date'
    # errors='ignore' 意思是：如果本来就没有这一列，也不要报错，忽略就好
    cols_to_remove = ['日期', 'time', 'system:index', '.geo'] 
    merged = merged.drop(columns=cols_to_remove, errors='ignore')

    # 将除了 Date 和 County 以外的列放在后面，稍微排个序让 Date 在前面（选做，看着舒服点）
    cols = ['Date', 'County'] + [c for c in merged.columns if c not in ['Date', 'County']]
    merged = merged[cols]

    # --- 第六步：装箱 ---
    all_years_data.append(merged)
    print(f"  成功！这一年合并后有 {len(merged)} 行数据。")

# ================= 收尾 =================
print("\n正在把所有年份的数据粘在一起...")

if all_years_data:
    # concat 就是把箱子里的表格，像积木一样上下堆起来
    final_result = pd.concat(all_years_data, ignore_index=True)
    
    # 保存结果
    save_path = os.path.join(dir_veg, "2001-2019_Total_Merged.csv")
    final_result.to_csv(save_path, index=False, encoding='gbk')
    
    print("-" * 30)
    print(f"大功告成！！")
    print(f"总共合并了 {len(final_result)} 行数据。")
    print(f"文件已保存在：{save_path}")
else:
    print("奇怪，没有合并成功任何数据，请检查文件名或路径。")