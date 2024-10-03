def extract_fields_from_line(line, field_indices):
    """从逗号分隔的行中提取指定索引的字段，并用空格连接这些字段。"""
    fields = line.strip().split(',')
    selected_fields = [fields[i] for i in field_indices if i < len(fields)]
    return ' '.join(selected_fields)

def merge_files(output_file, input_file1, field_indices1, input_file2, field_indices2, input_file3, field_indices3):
    """将多个文件中的指定字段合并到一个输出文件中，字段之间用空格隔开。"""
    # 读取所有输入文件的内容
    with open(input_file1, 'r') as infile1, open(input_file2, 'r') as infile2, open(input_file3, 'r') as infile3:
        lines1 = infile1.readlines()
        lines2 = infile2.readlines()
        lines3 = infile3.readlines()

    # 处理文件内容
    with open(output_file, 'w') as outfile:
        for line1, line2, line3 in zip(lines1[1:], lines2[1:], lines3[1:]):
            value1 = extract_fields_from_line(line1, field_indices1)
            value2 = extract_fields_from_line(line2, field_indices2)
            value3 = extract_fields_from_line(line3, field_indices3)
            if value1 and value2 and value3:
                value = f"{value1} {value2} {value3} 0 0 0 0"
                outfile.write(value + '\n')

def main():
    """主函数，执行文件合并操作。"""
    # 输出文件路径
    output_file = '/home/ssx/shengshuxuan/datasets/bit/bit_compus_1/merged_output.txt'
    # 文件路径列表
    input_file1 = '/home/ssx/shengshuxuan/datasets/bit/bit_compus_1/inspvax.txt'
    input_file2 = '/home/ssx/shengshuxuan/datasets/bit/bit_compus_1/imu.txt'
    # 需要提取的字段索引列表
    field_indices1 = [13, 14, 15, 20, 21, 22, 17, 18, 17, 18, 19] # lat、lon、height、roll、pitch、yaw、vel_north、vel_east、vel_north、vel_east、vel_up
    field_indices2 = [29, 30, 31, 29, 30, 31, 17, 18, 19, 17, 18, 19] # acc_x、acc_y、acc_z、acc_x、acc_y、acc_z、gyr_x、gyr_y、gyr_z、gyr_x、gyr_y、gyr_z
    field_indices3 = [23, 26, 11] #latitude_stdev、north_velocity_stdev、ins_status.status、
    merge_files(output_file, input_file1, field_indices1, input_file2, field_indices2, input_file1, field_indices3)
    print(f"已从文件合并到'{output_file}'")

if __name__ == "__main__":
    main()
