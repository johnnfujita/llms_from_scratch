import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)


query = inputs[1]  # 1
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)


# attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())


attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = inputs[1]  # 1

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    print(attn_weights_2[i])
    print(x_i)
    print(attn_weights_2[i] * x_i)
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

## This matrix mul with loop is slow
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)
## USE the @ with row @ row.TRANSPOSED
attn_scores = inputs @ inputs.T

print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
## SETTING TO -1 dim
# By setting dim=-1,
# we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor.
# If attn_scores is a two-dimensional tensor (for example, with a shape of [rows, columns]),
# it will normalize across the columns so that the values in each row
# (summing over the column dimension) sum up to 1.

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

## DOT product demo
# res = 0.0
# for idx, element in enumerate(inputs[0]):
#     print(f"\ninput -> {inputs[0][idx]} * ")
#     print(f"query -> {query[idx]}")
#     print(f"\n= {inputs[0][idx] * query[idx]}")
#     res += inputs[0][idx] * query[idx]
#     print(f"\nsum: {res}")
# print(res)
# print(torch.dot(inputs[0], query))
