import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
from data import get_BDIC

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')  # 可选，Mac 上推荐显示指定


# data = get_BDIC(handler)


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=200, help="number of init labeled samples")#初始10000
    parser.add_argument('--n_query', type=int, default=200, help="number of queries per round")#初始1000
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="FashionMNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10","BDIC"], help="dataset")
    parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                        choices=["RandomSampling",
                                 "LeastConfidence",
                                 "MarginSampling",
                                 "EntropySampling",
                                 "LeastConfidenceDropout",
                                 "MarginSamplingDropout",
                                 "EntropySamplingDropout",
                                 "KMeansSampling",
                                 "KCenterGreedy",
                                 "BALDDropout",
                                 "AdversarialBIM",
                                 "AdversarialDeepFool"], help="query strategy")
    args = parser.parse_args()
    pprint(vars(args))
    print()

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = get_dataset(args.dataset_name)                   # load dataset加载数据集（MNIST）
    net = get_net(args.dataset_name, device)                   # load network 加载与数据集匹配的神经网络
    strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy 获取选定的主动学习策略实例（比如随机采样）

    # start experiment
    dataset.initialize_labels(args.n_init_labeled) #随机选取 n_init_labeled 个样本作为初始的有标签样本
    # init_idxs = np.load('./data/init_shuffle_idxs.npy')[:args.n_init_labeled]#加载固定文件的有标签样本
    # dataset.initialize_labels_with_idxs(init_idxs)

    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # round 0 accuracy
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

    # 使用的 9 种不同采样策略（按顺序）
    #Macos最好在跑之前输入 export MallocStackLogging=0（不影响结果）
    strategy_names = [
        "KMeansSampling",
        "LeastConfidence",
        "MarginSamplingDropout",
        "KCenterGreedy",
        "EntropySampling",
        "BALDDropout",
        "AdversarialBIM",
        "AdversarialDeepFool",
        "LeastConfidence"  # 最后一轮强化边界学习
    ]

    # Round 0 初始化已完成，从 Round 1 开始迭代
    for rd in range(1, args.n_round):
        print(f"\n=== Round {rd} ===")

        # 根据当前轮次选择策略名
        strategy_name = strategy_names[rd - 1]
        print(f"Using strategy: {strategy_name}")

        # 重新实例化一个新策略（使用统一的 get_strategy 工具函数）
        strategy = get_strategy(strategy_name)(dataset, net)

        # 选出新的 query 样本索引
        query_idxs = strategy.query(args.n_query)

        # 将选中的样本加入 labeled set
        strategy.update(query_idxs)

        # 用更新后的 labeled set 重新训练模型
        strategy.train()

        # 在 test set 上进行预测并评估精度
        preds = strategy.predict(dataset.get_test_data())
        acc = dataset.cal_test_acc(preds)
        print(f"Round {rd} testing accuracy: {acc:.4f}")


    #注释掉的这部分是老师原代码
    #先改策略吧 先问老师数据集怎么改
    # for rd in range(1, args.n_round+1):
    #     print(f"Round {rd}")
    #
    #     # query
    #     query_idxs = strategy.query(args.n_query)
    #
    #     # update labels
    #     strategy.update(query_idxs)
    #     strategy.train()
    #
    #     # calculate accuracy
    #     preds = strategy.predict(dataset.get_test_data())
    #     print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")