import pandas as pd

def extract_strong_binders(netmhcpan_output_file):
    """
    解析 NetMHCpan 结果文件，提取强结合 (Strong Binder, SB) 的肽段及其相关属性。
    """
    sb_data = []
    
    with open(netmhcpan_output_file, 'r') as file:
        for line in file:
            line = line.strip()
            # 锁定被标记为强结合 (Strong Binder) 的行
            if line.endswith('<= SB'):
                parts = line.split()
                
                # 根据 NetMHCpan EL+BA 模式的输出格式解析对应列
                pos = parts[0]
                mhc = parts[1]
                peptide = parts[2]
                core = parts[3]
                
                # 中间的 Gp, Gl, Ip, Il, Icore 等属性跳过
                identity = parts[10]
                
                # 提取 Identity 之后的评分属性
                score_el = parts[11]
                rank_el = parts[12]
                score_ba = parts[13]
                rank_ba = parts[14]
                aff_nm = parts[15]
                
                sb_data.append([
                    pos, mhc, peptide, core, identity, 
                    score_el, rank_el, score_ba, rank_ba, aff_nm, 'SB'
                ])
                
    # 指定提取的列名
    columns = [
        'Pos', 'MHC', 'Peptide', 'Core', 'Identity', 
        'Score_EL', '%Rank_EL', 'Score_BA', '%Rank_BA', 'Aff(nM)', 'BindLevel'
    ]
    
    # 转换为 DataFrame
    df_sb = pd.DataFrame(sb_data, columns=columns)
    
    # 将数值类型的列转换为 float/int，方便后续进行统计过滤或可视化（如绘制亲和力分布的 density plot）
    numeric_cols = ['Pos', 'Score_EL', '%Rank_EL', 'Score_BA', '%Rank_BA', 'Aff(nM)']
    df_sb[numeric_cols] = df_sb[numeric_cols].apply(pd.to_numeric)
    
    return df_sb