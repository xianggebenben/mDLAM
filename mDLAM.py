# coding: utf-8
import numpy as np
import torch
import sys
import time
from torch.nn.functional import relu

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

from input_data import cora, pubmed, citeseer,coauthor_cs

dataset = cora()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp = torch.exp(x)
    imask = torch.eq(exp, float("inf"))
    exp = torch.where(imask, torch.exp(torch.tensor(88.6)) * torch.ones(size=exp.size()), exp)
    return exp / (torch.sum(exp, dim=0) + 1e-10)


def cross_entropy(label, prob):
    loss = -torch.sum(label * torch.log(prob + 1e-10))
    return loss


def cross_entropy_with_softmax(label, z):
    prob = softmax(z)
    loss = cross_entropy(label, prob)
    return loss


def Net(images, label, num_of_neurons):
    seed_num = 0
    torch.manual_seed(seed=seed_num)
    W1 = torch.normal(0, 0.1, size=(num_of_neurons, images.shape[0]))
    # W1=torch.ones((num_of_neurons, 28 * 28))
    torch.manual_seed(seed=seed_num)
    b1 = torch.normal(0, 0.1, size=(num_of_neurons, 1))
    # b1 =torch.ones((num_of_neurons, 1))
    z1 = torch.matmul(W1, images) + b1
    a1 = relu(z1)
    torch.manual_seed(seed=seed_num)
    W2 = torch.normal(0, 0.1, size=(num_of_neurons, num_of_neurons))
    # W2 = torch.ones((num_of_neurons, num_of_neurons))
    torch.manual_seed(seed=seed_num)
    b2 = torch.normal(0, 0.1, size=(num_of_neurons, 1))
    z2 = torch.matmul(W2, a1) + b2
    a2 = relu(z2)
    torch.manual_seed(seed=seed_num)
    W3 = torch.normal(0, 0.1, size=(dataset.num_classes, num_of_neurons))
    torch.manual_seed(seed=seed_num)
    b3 = torch.normal(0, 0.1, size=(dataset.num_classes, 1))
    z3 = torch.matmul(W3, a2) + b3
    # a3 = relu(z3)
    # torch.random.seed(seed=seed_num)
    # W4 = torch.normal(0, 0.1, size=(num_of_neurons, num_of_neurons))
    # torch.random.seed(seed=seed_num)
    # b4 = torch.normal(0, 0.1, size=(num_of_neurons, 1))
    # z4 = torch.matmul(W4, a3) + b4
    # a4 = relu(z4)
    # torch.random.seed(seed=seed_num)
    # W5 = torch.normal(0, 0.1, size=(10, num_of_neurons))
    # torch.random.seed(seed=seed_num)
    # b5 = torch.normal(0, 0.1, size=(10, 1))
    z3 = torch.ones(label.shape)
    z3[label == 0] = -1
    z3[label == 1] = 1
    return W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3


# In[2]:
def phi(a, W_next, b_next, z_next, rho):
    temp = z_next - torch.matmul(W_next, a) - b_next
    res = rho / 2 * torch.sum(temp * temp) + mu / 2 * torch.norm(W_next) * torch.norm(W_next)
    return res


# In[3]:


def phi_a(a, W_next, b_next, z_next, rho):
    res = rho * torch.matmul(torch.transpose(W_next, 0, 1), torch.matmul(W_next, a) + b_next - z_next)
    return res


def phi_W(a, W_next, b_next, z_next, rho, mu):
    temp = torch.matmul(W_next, a) + b_next - z_next
    temp2 = a.T
    res = rho * torch.matmul(temp, temp2) + mu * W_next
    return res


def phi_b(a, W_next, b_next, z_next, rho):
    res = torch.mean(rho * (torch.matmul(W_next, a) + b_next - z_next), dim=1).reshape(-1, 1)
    return res


def phi_z(a, W_next, b_next, z_next, rho):
    res = rho * (z_next - b_next - torch.matmul(W_next, a))
    return res


# In[4]:


def P(W_new, theta, a_last, W, b, z, rho, mu):
    temp = W_new - W
    res = phi(a_last, W, b, z, rho) + mu / 2 * torch.norm(W_new) * torch.norm(W_new) + torch.sum(
        phi_W(a_last, W, b, z, rho, mu) * temp) + torch.sum(theta * temp * temp) / 2
    return res


# In[5]:


def Q(a_new, tau, a, W_next, b_next, z_next, rho):
    # tf = ~torch.isinf(tau)
    temp = a_new - a
    res = phi(a, W_next, b_next, z_next, rho) + torch.sum(phi_a(a, W_next, b_next, z_next, rho) * temp) + torch.sum(
        tau * temp * temp) / 2
    return res


def update_W(a_last, b, z, W_old, rho, mu):
    gradients = phi_W(a_last, W_old, b, z, rho, mu)
    gamma = 2
    alpha = 1
    zeta = W_old - gradients / alpha
    count = 0
    while (phi(a_last, zeta, b, z, rho) + mu / 2 * torch.norm(zeta) * torch.norm(zeta) > P(zeta, alpha, a_last, W_old,
                                                                                           b, z, rho, mu)):
        alpha = alpha * gamma
        zeta = W_old - gradients / alpha  # Learning rate decreases to 0, leading to infinity loop here.
        count += 1
        if count > 10:
            zeta = W_old
            break
    theta = alpha
    W = zeta
    return W


def update_b(a_last, W, z, b_old, rho):
    gradients = phi_b(a_last, W, b_old, z, rho)
    res = b_old - gradients / rho
    return res


def update_z(a_last, W, b, a, eps, z_old, rho):
    gradients = phi_z(a_last, W, b, z_old, rho)
    tolerance = 10e-10
    temp = a - eps >= tolerance
    z = torch.minimum(z_old - gradients / rho, a + eps - tolerance)
    z[temp] = torch.maximum(z[temp], a[temp] - eps + tolerance)
    # z = torch.minimum(z, a - eps)
    return z


def update_zl(a_last, W, b, label, zl_old, rho):
    gradients1 = phi_z(a_last, W, b, zl_old, rho)
    fzl = 10e10
    MAX_ITER = 500
    zl = zl_old
    lamda = 1
    zeta = zl
    eta = 4
    TOLERANCE = 10e-5
    for i in range(MAX_ITER):
        fzl_old = fzl
        temp = zl - (zl_old - gradients1 / rho)
        fzl = cross_entropy_with_softmax(label, zl) + rho / 2 * torch.sum(temp * temp)
        if abs(fzl - fzl_old) < TOLERANCE:
            break
        lamda_old = lamda
        lamda = (1 + np.sqrt(1 + 4 * lamda * lamda)) / 2
        gamma = (1 - lamda_old) / lamda
        gradients2 = (softmax(zl) - label)
        zeta_old = zeta
        zeta = (rho * (zl_old - gradients1 / rho) + (zl - eta * gradients2) / eta) / (rho + 1 / eta)
        zl = (1 - gamma) * zeta + gamma * zeta_old
    return zl


def update_a(W_next, b_next, z_next, z, eps, a_old, rho):
    tolerance = 10e-10
    Relu = relu(z)
    gradients = phi_a(a_old, W_next, b_next, z_next, rho)
    eta = 2
    up_bound = Relu + eps
    lo_bound = Relu - eps
    t = 1
    beta = a_old - gradients / t
    beta = torch.maximum(torch.minimum(beta, up_bound - tolerance), lo_bound + tolerance)
    count = 0
    while (phi(beta, W_next, b_next, z_next, rho) > Q(beta, t, a_old, W_next, b_next, z_next, rho)):
        t = t * eta
        beta = a_old - gradients / t  # Learning rate decreases to 0, leading to infinity loop here.
        beta = torch.maximum(torch.minimum(beta, up_bound - tolerance), lo_bound + tolerance)
        count += 1
        if count > 10:
            beta = a_old
            break
    tau = t
    a = beta
    return a


def test_accuracy(W1, b1, W2, b2, W3, b3, images, labels):
    nums = labels.shape[1]
    z1 = torch.matmul(W1, images) + b1
    a1 = relu(z1)
    z2 = torch.matmul(W2, a1) + b2
    a2 = relu(z2)
    z3 = torch.matmul(W3, a2) + b3
    cost = cross_entropy_with_softmax(labels, z3) / nums
    pred = torch.argmax(labels, dim=0)
    label = torch.argmax(z3, dim=0)
    # print(pred, "\n", label)
    return (torch.sum(torch.eq(pred, label), dtype=torch.float32).item() / nums, cost)


def objective(x,W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3,y):
    loss = cross_entropy_with_softmax(y, z3)
    r1 = phi(x, W1, b1, z1, rho)
    r2 = phi(a1, W2, b2, z2, rho)
    r3 = phi(a2, W3, b3, z3, rho)
    res = loss + r1 + r2 + r3 + mu / 2 * torch.norm(W1) * torch.norm(W1) + mu / 2 * torch.norm(W2) * torch.norm(
        W2) + mu / 2 * torch.norm(W3) * torch.norm(W3)
    return res


x_train = torch.transpose(dataset.x_train, 0, 1)
y_train = torch.transpose(dataset.y_train, 0, 1)

x_test = torch.transpose(dataset.x_test, 0, 1)
y_test = torch.transpose(dataset.y_test, 0, 1)

num_of_neurons = 100

sample_num = x_train.shape[1]
# print(all_samples)
ITER = 200
index = 0
train_acc = np.zeros(ITER)
test_acc = np.zeros(ITER)
objective_value = np.zeros(ITER)
train_cost = np.zeros(ITER)
test_cost = np.zeros(ITER)
gap = 0.
W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3 = Net(x_train, y_train, num_of_neurons)
eps = 100
rho = 0.001
mu = 0.05
W1_old = W1
b1_old = b1
z1_old = z1
a1_old = a1
W2_old = W2
b2_old = b2
z2_old = z2
a2_old = a2
W3_old = W3
b3_old = b3
z3_old = z3
t_old = 0
for i in range(ITER):  # 1
    pre = time.time()
    t = (1 + np.sqrt(1 + 4 * t_old * t_old)) / 2
    W1_old = W1
    W1 = W1 + (t_old - 1) / t * (W1 - W1_old)
    W1 = update_W(x_train, b1, z1, W1, rho, mu)
    if(objective(x_train,W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3,y_train)>objective(x_train,W1_old, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3,y_train)):
        W1 = update_W(x_train, b1, z1, W1_old, rho, mu)
    b1_old = b1
    b1 = b1 + (t_old - 1) / t * (b1 - b1_old)
    b1 = update_b(x_train, W1, z1, b1, rho)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1_old, z1, a1,
                                                                                            W2, b2, z2, a2, W3, b3, z3,
                                                                                            y_train)):
        b1 = update_b(x_train, W1, z1, b1_old, rho)
    z1_old = z1
    z1 = z1 + (t_old - 1) / t * (z1 - z1_old)
    z1 = update_z(x_train, W1, b1, a1, eps, z1, rho)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1_old, a1,
                                                                                            W2, b2, z2, a2, W3, b3, z3,
                                                                                            y_train)):
        z1 = update_z(x_train, W1, b1, a1, eps, z1_old, rho)
    a1_old = a1
    a1 = a1 + (t_old - 1) / t * (a1 - a1_old)
    a1 = update_a(W2, b2, z2, z1, eps, a1, rho)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1, a1_old,
                                                                                            W2, b2, z2, a2, W3, b3, z3,
                                                                                            y_train)):
        a1 = update_a(W2, b2, z2, z1, eps, a1_old, rho)
    gap1 = np.power(torch.linalg.norm(a1 - relu(z1)), 2)
    W2_old = W2
    W2 = W2 + (t_old - 1) / t * (W2 - W2_old)
    W2 = update_W(a1, b2, z2, W2, rho, mu)
    if(objective(x_train,W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3,y_train)>objective(x_train,W1, b1, z1, a1, W2_old, b2, z2, a2, W3, b3, z3,y_train)):
        W2 = update_W(a1, b2, z2, W2_old, rho, mu)
    b2_old = b2
    b2 = b2 + (t_old - 1) / t * (b2 - b2_old)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1, a1,
                                                                                            W2, b2_old, z2, a2, W3, b3, z3,
                                                                                            y_train)):
        b2 = update_b(a1, W2, z2, b2_old, rho)
    z2_old = z2
    z2 = z2 + (t_old - 1) / t * (z2 - z2_old)
    z2 = update_z(a1, W2, b2, a2, eps, z2, rho)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1, a1,
                                                                                            W2, b2, z2_old, a2, W3, b3, z3,
                                                                                            y_train)):
        z2 = update_z(a1, W2, b2, a2, eps, z2_old, rho)
    a2_old = a2
    a2 = a2 + (t_old - 1) / t * (a2 - a2_old)
    a2 = update_a(W3, b3, z3, z2, eps, a2, rho)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1, a1_old,
                                                                                            W2, b2, z2, a2_old, W3, b3, z3,
                                                                                            y_train)):
        a2 = update_a(W3, b3, z3, z2, eps, a2_old, rho)
    gap2 = np.power(torch.linalg.norm(a2 - relu(z2)), 2)
    W3_old = W3
    W3 = W3 + (t_old - 1) / t * (W3 - W3_old)
    W3 = update_W(a2, b3, z3, W3, rho, mu)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1, a1,
                                                                                            W2, b2, z2, a2, W3_old, b3,
                                                                                            z3, y_train)):
        W3 = update_W(a2, b3, z3, W3_old, rho, mu)
    b3_old = b3
    b3 = b3 + (t_old - 1) / t * (b3 - b3_old)
    b3 = update_b(a2, W3, z3, b3, rho)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1, a1,
                                                                                            W2, b2, z2, a2, W3_old, b3,
                                                                                            z3,
                                                                                            y_train)):
        b3 = update_b(a2, W3, z3, b3_old, rho)
    b3 = update_b(a2, W3, z3, b3, rho)
    z3_old = z3
    z3 = z3 + (t_old - 1) / t * (z3 - z3_old)
    z3 = update_zl(a2, W3, b3, y_train, z3, rho)
    if (objective(x_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, y_train) > objective(x_train, W1, b1, z1, a1,
                                                                                            W2, b2, z2, a2, W3_old, b3,
                                                                                            z3,
                                                                                            y_train)):
        z3 = update_zl(a2, W3, b3, y_train, z3_old, rho)
    after = time.time()
    gap = gap1 + gap2
    obj1 = objective(x_train,W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3,y_train)
    objective_value[i]=obj1
    obj2 = obj1 + gap1 + gap2
    print("eps=", eps)
    print("Iteration ", i)
    print("gap=", gap.numpy())
    print("obj1=", obj1.numpy())
    print("obj2=", obj2.numpy())
    train_acc[i], train_cost[i] = test_accuracy(W1, b1, W2, b2, W3, b3, x_train, y_train)
    test_acc[i], test_cost[i] = test_accuracy(W1, b1, W2, b2, W3, b3, x_test, y_test)
    objective_value[i] = obj1.numpy()
    print("training accuracy is", train_acc[i])
    print("training cost is", train_cost[i])
    print("test accuracy is", test_acc[i])
    print("test cost is", test_cost[i])
    print("training time:", after - pre)
    eps=max(eps/2,0.001)
torch.save(
        {"obj":objective_value, "train_acc": train_acc, "train_cost": train_cost, "test_acc": test_acc, "test_cost": test_cost},
        'mDLAM_cora' + repr(num_of_neurons) + '_3layers.pt')



