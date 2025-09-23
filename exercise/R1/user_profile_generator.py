import pandas as pd
import numpy as np
import yaml

def generate_user_profiles(config_path: str):
    """
    根据配置文件生成用户画像数据。

    :param config_path: 配置文件路径 (YAML格式)
    """
    # 1. 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    user_count = config['USER_COUNT']
    seed = config['SEED']
    output_file = config['output_file']

    # 设置随机种子
    np.random.seed(seed)

    # 2. 生成类别特征
    user_data = {}
    for feature, params in config['categorical_features'].items():
        user_data[feature] = np.random.choice(
            params['choices'],
            size=user_count,
            p=params['probabilities']
        )

    # 3. 生成数值特征
    for feature, params in config['numerical_features'].items():
        if params['type'] == 'normal':
            sample = np.random.normal(
                loc=params['loc'],
                scale=params['scale'],
                size=user_count
            )
        elif params['type'] == 'exponential':
            sample = np.random.exponential(
                scale=params['scale'],
                size=user_count
            )
        else:
            raise ValueError(f"不支持的数值特征类型: {params['type']}")

        # 限制取值范围
        user_data[feature] = np.clip(
            sample,
            params['clip_min'],
            params['clip_max']
        ).astype(int)

    # 4. 合成DataFrame
    user_df = pd.DataFrame(user_data)

    # 5. 保存为CSV
    user_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 已生成 {user_count} 条用户画像数据，已保存到 '{output_file}' 文件。")


if __name__ == '__main__':
    # 使用 profile.yaml 配置文件生成数据
    generate_user_profiles('exercise/R1/profile.yaml')