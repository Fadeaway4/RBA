"""
From https://github.com/vikram2000b/bad-teaching-unlearning / https://arxiv.org/abs/2205.08096
"""

from torch.nn import functional as F
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def JSDiv(p, q):
    m = (p + q) / 2
    # return 0.5 * F.kl_div(torch.log(p), m, reduction='batchmean') + 0.5 * F.kl_div(torch.log(q), m, reduction='batchmean')
    return 0.5 * F.kl_div(torch.log(m), p, reduction='batchmean') + 0.5 * F.kl_div(torch.log(m), q, reduction='batchmean')



# ZRF/UnLearningScore https://arxiv.org/abs/2205.08096
def UnLearningScore(tmodel, gold_model, forget_dl, batch_size, device):
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for batch in forget_dl:
            x, y, cy = batch
            x = x.to(device)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    return 1 - JSDiv(model_preds, gold_model_preds)


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, _, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    # print("retain_prob", len(retain_prob))
    # print("forget_prob", len(forget_prob))
    # print("test_prob", len(test_prob))

    X_r = (torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1))
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])

    return X_f, Y_f, X_r, Y_r

# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", multi_class="multinomial")
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

def get_membership_attack_prob_strategy_b(forget_train_loader, forget_test_loader, model):
    """
    策略 B：严格成员推断攻击
    只区分：遗忘类的训练集 (Member) vs 遗忘类的测试集 (Non-member)
    """
    model.eval()
    # 1. 收集熵特征
    # 注意：这里的 forget_test_loader 必须只包含遗忘类的测试数据
    forget_train_prob = collect_prob(forget_train_loader, model)
    forget_test_prob = collect_prob(forget_test_loader, model)

    X_member = entropy(forget_train_prob).cpu().numpy().reshape(-1, 1)
    Y_member = np.ones(len(forget_train_prob))

    X_nonmember = entropy(forget_test_prob).cpu().numpy().reshape(-1, 1)
    Y_nonmember = np.zeros(len(forget_test_prob))

    X = np.concatenate([X_member, X_nonmember])
    Y = np.concatenate([Y_member, Y_nonmember])

    # 2. 划分攻击模型的训练集和测试集 (4:6 划分)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.6, random_state=42, stratify=Y
        )
    except ValueError:
        # 处理样本量极少的情况
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.5, random_state=42
        )

    # 3. 训练攻击者
    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X_train, y_train)

    # 4. 评估指标
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return acc, auc

@torch.no_grad()
def actv_dist(model1, model2, dataloader, device="cuda"):
    sftmx = torch.nn.Softmax(dim=1)
    distances = []
    for batch in dataloader:
        x, _, _ = batch
        x = x.to(device)
        model1_out = model1(x)
        model2_out = model2(x)
        diff = torch.sqrt(
            torch.sum(
                torch.square(
                    F.softmax(model1_out, dim=1) - F.softmax(model2_out, dim=1)
                ),
                axis=1,
            )
        )
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis=0)
    return distances.mean()
