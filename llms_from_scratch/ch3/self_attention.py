from ch3.self_attention_v1 import SelfAttention_v1
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

x_2 = inputs[1]  # 1
d_in = inputs.shape[1]  # 2
d_out = 2  # 3
print("x_2:", x_2)
print("d_in:", d_in)
print("d_out:", d_out)

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("W_query:", W_query)
print("W_key:", W_key)
print("W_value:", W_value)

# 1 x 3 @ 3 x 2 = 1 x 2
query_2 = x_2 @ W_query
# 1 x 3 @ 3 x 2 = 1 x 2
key_2 = x_2 @ W_key
# 1 x 3 @ 3 x 2 = 1 x 2
value_2 = x_2 @ W_value

print("Query vector:", query_2)
print("Key vector:", key_2)
print("Value vector:", value_2)


keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query

print("Queries matrix:", queries)
print("Keys matrix:", keys)
print("Values matrix:", values)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


keys_2 = keys[1]  # 1
print("query_2:", query_2)
print("keys_2:", keys_2)

attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# 1 x 2 @ 2 x 6 = 1 x 6
# tensor([0.4306, 1.4551]) @
# tensor([[0.3669, 0.4433, 0.4361, 0.2408, 0.1827, 0.3275],
#         [0.7646, 1.1419, 1.1156, 0.6706, 0.3292, 0.9642]])
# column tensor([
# 0.4306*0.3669 + 1.4551*0.7646 = 1.2705,
# 0.4306*0.4433 + 1.4551*1.1419 = 1.8524,
# 0.4306*0.4361 + 1.4551*1.1156 = 1.8111,
# 0.4306*0.2408 + 1.4551*0.6706 = 1.0795,
# 0.4306*0.1827 + 1.4551*0.3292 = 0.5577,
# 0.4306*0.3275 + 1.4551*0.9642 = 1.5440])
attn_scores_2 = query_2 @ keys.T  # 1
print(attn_scores_2)

d_k = keys.shape[-1]
print("d_k", d_k)
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(f"scalled scores {attn_scores_2 / d_k**0.5}")
print(attn_weights_2)

## attn_weights_2
# 1 X 6
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
# values
# 6 X 2
# tensor[0.1855, 0.8812],
# [0.3951, 1.0037],
# [0.3879, 0.9831],
# [0.2393, 0.5493],
# [0.1492, 0.3346],
# [0.3221, 0.7863]])
# context_vec_2
# 1 X 2
# tensor([0.3061, 0.8210])
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

attn_scores = queries @ keys.T  # omega
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
context_vec = attn_weights @ values
print("Attention scores:", attn_scores)
print("Attention weights:", attn_weights)
print("Context vector:", context_vec)
print("Context vector shape:", context_vec.shape)


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
