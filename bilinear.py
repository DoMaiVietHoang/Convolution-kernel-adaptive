import torch

# 1. Giả lập ảnh đầu vào: batch=1, channel=1, H=5, W=5
x = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
print("Ảnh đầu vào:\n", x)

# 2. Điểm cần lấy mẫu: (2.3, 3.6) — tọa độ thực
p = torch.tensor([[[[2.3, 3.6]]]])  # shape (1, 1, 1, 2)
N = 1  # chỉ lấy 1 điểm

# 3. Tính 4 điểm lân cận (q_lt, q_rb, q_lb, q_rt)
q_lt = p.floor()
q_rb = q_lt + 1
q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

# 4. Clamp lại để nằm trong ảnh
H, W = x.size(2), x.size(3)
def clamp_q(q):
    return torch.cat([
        torch.clamp(q[..., :N], 0, H - 1),
        torch.clamp(q[..., N:], 0, W - 1)
    ], dim=-1).long()

q_lt = clamp_q(q_lt)
q_rb = clamp_q(q_rb)
q_lb = clamp_q(q_lb)
q_rt = clamp_q(q_rt)

# 5. Hàm lấy giá trị tại tọa độ q
def get_x_q(x, q):
    b, c, h, w = x.size()
    flat = x.view(b, c, -1)
    idx = q[..., 0] * w + q[..., 1]  # flatten index
    idx = idx.view(b, 1, -1).expand(-1, c, -1)
    return flat.gather(-1, idx).view(b, c, 1, 1, N)

x_q_lt = get_x_q(x, q_lt)
x_q_rb = get_x_q(x, q_rb)
x_q_lb = get_x_q(x, q_lb)
x_q_rt = get_x_q(x, q_rt)

# 6. Tính trọng số bilinear
p_clamped = torch.cat([
    torch.clamp(p[..., :N], 0, H - 1),
    torch.clamp(p[..., N:], 0, W - 1)
], dim=-1)

def get_g(q):
    return (1 - torch.abs(q[..., :N].float() - p_clamped[..., :N])) * \
           (1 - torch.abs(q[..., N:].float() - p_clamped[..., N:]))

g_lt = get_g(q_lt)
g_rb = get_g(q_rb)
g_lb = get_g(q_lb)
g_rt = get_g(q_rt)

# 7. Tổng hợp nội suy
x_interp = g_lt.unsqueeze(1) * x_q_lt + \
           g_rb.unsqueeze(1) * x_q_rb + \
           g_lb.unsqueeze(1) * x_q_lb + \
           g_rt.unsqueeze(1) * x_q_rt

print("Giá trị nội suy tại (2.3, 3.6):", x_interp.item())
