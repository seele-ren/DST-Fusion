import pandas as pd

# 输入和输出文件路径
xlsx_file = "data/jilin_phenology_2001-2020.xlsx"
csv_file = "data/jilin_phenology_2001-2020.csv"

# 读取 Excel（默认读取第一个工作表）
df = pd.read_excel(xlsx_file)

# 保存为 CSV
df.to_csv(csv_file, index=False, encoding="utf-8-sig")

print("转换完成！")
