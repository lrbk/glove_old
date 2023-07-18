import csv
# 输入文件路径
input_file = 'data_l2.csv'
# 输出文件路径
output_file = 'data_new.csv'

# 读取并写入数据
with open(input_file, 'r', newline='') as csv_infile, \
        open(output_file, 'w', newline='') as csv_outfile:
    reader = csv.reader(csv_infile)
    writer = csv.writer(csv_outfile)
    # 读取并写入特定行
    rows_to_extract = range(100,200)  
    for i, row in enumerate(reader):
        if i in rows_to_extract:
            writer.writerow(row)
