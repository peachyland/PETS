def print_unique_questions(comparison_df):
    """
    专门用于打印只存在于其中一个数据集的问题
    """
    print("\n" + "="*50)
    print("🔍 独有题目分析 (Dataset Discrepancy)")
    print("="*50)

    # 1. 只在 lchen001 中存在的题目
    only_lchen = comparison_df[comparison_df['_merge'] == 'left_only']
    if not only_lchen.empty:
        print(f"\n📍 仅存在于 [lchen001/AIME2025] 的题目 ({len(only_lchen)} 个):")
        for i, row in enumerate(only_lchen['question'], 1):
            # 只打印前150个字符防止刷屏，可根据需要调整
            print(f"  {i}. {row[:150]}...")
    else:
        print("\n✅ 没有只存在于 lchen001 的题目。")

    # 2. 只在 opencompass 中存在的题目
    only_oc = comparison_df[comparison_df['_merge'] == 'right_only']
    if not only_oc.empty:
        print(f"\n📍 仅存在于 [opencompass/AIME2025] 的题目 ({len(only_oc)} 个):")
        for i, row in enumerate(only_oc['question'], 1):
            print(f"  {i}. {row[:150]}...")
    else:
        print("\n✅ 没有只存在于 opencompass 的题目。")
    
    print("\n" + "="*50)

# 在主程序中调用
# 假设 comparison 是上一步 merge 出来的结果：
# comparison = pd.merge(..., indicator=True)
# print_unique_questions(comparison)

from datasets import load_dataset
import pandas as pd

def run_full_comparison():
    # --- 加载数据 ---
    ds_l = load_dataset("lchen001/AIME2025", split='train').to_pandas()
    ds_l = ds_l.rename(columns={'Question': 'question', 'Answer': 'answer'})
    
    oc_list = []
    for sub in ["AIME2025-I", "AIME2025-II"]:
        try:
            temp = load_dataset("opencompass/AIME2025", sub, split='test').to_pandas()
            # 确保列名统一小写
            temp.columns = [c.lower() for c in temp.columns]
            oc_list.append(temp)
        except: pass
    ds_o = pd.concat(oc_list, ignore_index=True)

    # --- 清洗 ---
    for df in [ds_l, ds_o]:
        df['question'] = df['question'].astype(str).str.strip()
        df['answer'] = df['answer'].astype(str).str.strip()

    # --- 比对 ---
    comparison = pd.merge(
        ds_l[['question', 'answer']], 
        ds_o[['question', 'answer']], 
        on='question', 
        how='outer', 
        suffixes=('_lchen', '_oc'),
        indicator=True
    )

    # --- 打印独有题目 ---
    print_unique_questions(comparison)
    
    # --- 顺便打印答案不一致的 (如果有) ---
    mismatched = comparison[(comparison['_merge'] == 'both') & (comparison['answer_lchen'] != comparison['answer_oc'])]
    if not mismatched.empty:
        print(f"\n❌ 发现 {len(mismatched)} 个题目虽然共有，但答案不同！")
        print(mismatched[['question', 'answer_lchen', 'answer_oc']])

if __name__ == "__main__":
    run_full_comparison()