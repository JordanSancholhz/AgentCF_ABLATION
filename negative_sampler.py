"""
固定负样本生成器
从config.py导入所有路径，零硬编码
✅ 支持多数据集，负样本文件按数据集分类存储
"""

import random
import json
import pandas as pd
import argparse
import os

# ✅ 从config导入路径，零硬编码
from config import train_file, test_file, random_file, CURRENT_DATASET, DATASET_DIR

from dataPrepare import createRandomDF

def generate_fixed_train_negatives(seed):
    """生成训练时的固定负样本映射"""
    print(f"🔧 生成训练负样本（数据集={CURRENT_DATASET}, seed={seed}）...")
    random.seed(seed)
    
    # ✅ 使用config中的路径
    train_df = pd.read_csv(train_file, sep='\t', header=0)
    random_df = createRandomDF(random_file)
    
    train_negatives = {}
    stats = {"total_pairs": 0, "users": set(), "seed": seed, "dataset": CURRENT_DATASET}
    
    user_groups = train_df.groupby('user_id:token')
    
    for user_id, group in user_groups:
        user_id_str = str(user_id).strip()
        stats["users"].add(user_id_str)
        
        user_row = random_df[random_df['user_id'] == int(user_id)]
        if len(user_row) == 0:
            print(f"⚠️ 用户 {user_id} 在random文件中不存在")
            continue
        
        candidates = user_row['candidates'].values[0]
        
        for round_num, (idx, row) in enumerate(group.iterrows()):
            pos_item_id = str(row['item_id:token']).strip()
            valid_candidates = [c for c in candidates if c != pos_item_id]
            
            if len(valid_candidates) == 0:
                print(f"❌ 用户 {user_id} 的正样本 {pos_item_id} 没有可用负样本")
                continue
            
            neg_item_id = random.choice(valid_candidates)
            key = f"user_{user_id_str}_pos_{pos_item_id}_round_{round_num}"
            train_negatives[key] = neg_item_id
            stats["total_pairs"] += 1
    
    stats["users"] = len(stats["users"])
    
    # ✅ 修改：按数据集分类存储
    output_file = f"dataset/{DATASET_DIR}/train_negatives_seed{seed}.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"metadata": stats, "negatives": train_negatives}, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 训练负样本已保存: {output_file}")
    print(f"   正负样本对数: {stats['total_pairs']}, 用户数: {stats['users']}")
    
    return train_negatives

def generate_fixed_eval_candidates(seed, candidate_num):
    """生成评估时的固定候选集"""
    print(f"\n🔧 生成评估候选集（数据集={CURRENT_DATASET}, seed={seed}）...")
    random.seed(seed)
    
    # ✅ 使用config中的路径
    test_df = pd.read_csv(test_file, sep='\t', header=0)
    random_df = createRandomDF(random_file)
    
    eval_candidates = {}
    stats = {"total_users": 0, "candidate_num": candidate_num, "seed": seed, "dataset": CURRENT_DATASET}
    
    for idx, row in test_df.iterrows():
        user_id = str(row['user_id:token']).strip()
        target_item_id = str(row['item_id:token']).strip()
        
        user_row = random_df[random_df['user_id'] == int(user_id)]
        if len(user_row) == 0:
            continue
        
        candidates = user_row['candidates'].values[0]
        valid_candidates = [c for c in candidates if c != target_item_id]
        
        if len(valid_candidates) < (candidate_num - 1):
            print(f"⚠️ 用户 {user_id} 负样本不足")
            continue
        
        neg_samples = random.sample(valid_candidates, candidate_num - 1)
        candidate_list = neg_samples + [target_item_id]
        random.shuffle(candidate_list)
        
        eval_candidates[user_id] = {'target': target_item_id, 'candidates': candidate_list}
        stats["total_users"] += 1
    
    # ✅ 修改：按数据集分类存储
    output_file = f"dataset/{DATASET_DIR}/eval_candidates_seed{seed}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"metadata": stats, "candidates": eval_candidates}, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 评估候选集已保存: {output_file}")
    print(f"   用户数: {stats['total_users']}, 候选集大小: {candidate_num}")
    
    return eval_candidates

def verify_negatives(seed):
    """验证负样本正确性"""
    print(f"\n🔍 验证负样本（数据集={CURRENT_DATASET}）...")
    
    # ✅ 修改：使用按数据集分类的路径
    train_file_path = f"dataset/{DATASET_DIR}/train_negatives_seed{seed}.json"
    eval_file_path = f"dataset/{DATASET_DIR}/eval_candidates_seed{seed}.json"
    
    if not os.path.exists(train_file_path):
        print(f"❌ 训练负样本文件不存在: {train_file_path}")
        return False
    if not os.path.exists(eval_file_path):
        print(f"❌ 评估候选集文件不存在: {eval_file_path}")
        return False
    
    with open(train_file_path, 'r') as f:
        train_data = json.load(f)
    with open(eval_file_path, 'r') as f:
        eval_data = json.load(f)
    
    train_errors = 0
    for key, neg_item in train_data['negatives'].items():
        pos_item = key.split('_')[3]
        if neg_item == pos_item:
            print(f"❌ 训练负样本错误: {key}")
            train_errors += 1
    
    eval_errors = 0
    for user_id, data in eval_data['candidates'].items():
        target = data['target']
        candidates = data['candidates']
        
        if target not in candidates:
            print(f"❌ 用户{user_id}的target不在候选集中")
            eval_errors += 1
        if len(candidates) != len(set(candidates)):
            print(f"❌ 用户{user_id}有重复候选")
            eval_errors += 1
    
    if train_errors == 0 and eval_errors == 0:
        print(f"✅ 验证通过！")
        return True
    else:
        print(f"❌ 发现错误：训练{train_errors}个，评估{eval_errors}个")
        return False

if __name__ == "__main__":
    # ✅ 从config导入默认值
    from config import NEGATIVE_SAMPLE_SEED, candidate_num
    
    parser = argparse.ArgumentParser(description='生成固定负样本')
    parser.add_argument('--seed', type=int, default=NEGATIVE_SAMPLE_SEED, help='随机种子')
    parser.add_argument('--candidate_num', type=int, default=candidate_num, help='候选集大小')
    parser.add_argument('--verify', action='store_true', help='验证生成的负样本')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 固定负样本生成器")
    print(f"数据集: {CURRENT_DATASET}")
    print(f"Seed: {args.seed}, 候选集: {args.candidate_num}")
    print("=" * 60)
    
    generate_fixed_train_negatives(args.seed)
    generate_fixed_eval_candidates(args.seed, args.candidate_num)
    
    if args.verify:
        verify_negatives(args.seed)
    
    print("\n✅ 完成！")