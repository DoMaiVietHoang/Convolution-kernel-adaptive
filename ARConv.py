import torch
import torch.nn as nn

class ARConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, flag=False, modulation=True):
        super(ARConv, self).__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.hook_handles = []
        self.hook_handles.append(self.m_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.m_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[1].register_full_backward_hook(self._set_lr))
 
        self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)
 
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input
 
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()  # 移除钩子函数
        self.hook_handles.clear()  # 清空句柄列表
 
    def forward(self, x, epoch, hw_range):
        assert isinstance(hw_range, list) and len(hw_range) == 2, "hw_range should be a list with 2 elements, represent the range of h w"
        scale = hw_range[1] // 9
        if hw_range[0] == 1 and hw_range[1] == 3:
            scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w   the relationship between feture map and height and width
        w = self.w_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
        if epoch <= 100:
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)
            def phi(x):
                if x % 2 == 0:
                    x -= 1
                return x
            N_X, N_Y = phi(N_X), phi(N_Y)
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)
            N_X, N_Y = min(N_X, 7), min(N_Y, 7)
            if epoch == 100:
                self.reserved_NXY = self.reserved_NXY = nn.Parameter(
                    torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
                    requires_grad=False
                )
        else:
            N_X = self.reserved_NXY[0]
            N_Y = self.reserved_NXY[1]

        N = N_X * N_Y
        # print(N_X, N_Y)
        # Đoạn này xử lý giá trị pixel (l, w là các map giá trị, được lặp lại N lần để chuẩn bị 
        # cho việc tính offset toạ độ sampling)
        l = l.repeat([1, N, 1, 1])
        w = w.repeat([1, N, 1, 1])
        offset = torch.cat((l, w), dim=1)
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, N_X, N_Y)  # (b, 2*N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # clip p
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # (b, c, h, w, N)
        # Tại sao phải nội suy?
        # Vì các điểm sampling p thường không rơi đúng vào vị trí pixel integer trên feature map,
        # nên ta cần nội suy bilinear từ 4 điểm lân cận (q_lt, q_rb, q_lb, q_rt) để lấy giá trị tại p.
        # Kết quả của việc nội suy bilinear là giá trị đặc trưng (feature value) tại các vị trí sampling p,
        # được tính bằng tổng có trọng số (theo kernel bilinear) của 4 điểm lân cận (q_lt, q_rb, q_lb, q_rt).
        # x_offset chứa các giá trị feature đã được nội suy tại từng điểm sampling p trên feature map.
        # x_offset ở đây là feature map đã được nội suy bilinear từ 4 điểm lân cận,
        # giá trị của nó không phải là số nguyên mà là giá trị thực (float),
        # thể hiện đặc trưng tại vị trí sampling p trên feature map gốc.
        x_offset = (
                  g_lt.unsqueeze(dim=1) * x_q_lt
                + g_rb.unsqueeze(dim=1) * x_q_rb
                + g_lb.unsqueeze(dim=1) * x_q_lb
                + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        print(x_offset)
        x_offset = self.dropout2(x_offset)
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
        out = x_offset * m + bias
        return out
 
    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
 
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0
 
    def _get_p(self, offset, dtype, n_x, n_y):
        """
        Hàm này tính toán tọa độ sampling p cho từng điểm trong kernel biến thiên (adaptive kernel).

        Tham số:
            offset: tensor (b, 2N, h, w) - offset học được cho từng điểm sampling (N là số điểm sampling)
            dtype: kiểu dữ liệu torch (thường là torch.float32)
            n_x, n_y: số điểm sampling theo chiều x và y (kernel size)

        Các bước thực hiện:
        1. Tính N, h, w từ offset (N là số điểm sampling, h và w là kích thước feature map).
        2. Chia offset thành 2 phần: L (offset theo chiều x) và W (offset theo chiều y), mỗi phần có shape (b, N, h, w).
        3. Chuẩn hóa offset L và W theo kích thước kernel (chia cho n_x, n_y).
        4. Ghép lại thành offsett (b, 2N, h, w).
        5. Tạo p_n: vị trí tương đối của từng điểm sampling trong kernel (shape (1, 2N, 1, 1)), rồi lặp lại cho toàn bộ spatial (h, w).
        6. Tạo p_0: vị trí gốc của từng điểm sampling trên feature map (shape (1, 2N, h, w)).
        7. Tính p = p_0 + offsett * p_n: vị trí sampling cuối cùng (đã cộng offset).
        8. Trả về p.

        Kết quả trả về:
            p: tensor (b, 2N, h, w) - tọa độ sampling cuối cùng cho từng điểm kernel tại mỗi vị trí (h, w)
        """
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)  
        L, W = offset.split([N, N], dim=1)  # Devide feture map of L and W 
        L = L / n_x 
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p
 
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset
 
    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        """
        Hàm này 'làm rộng' (expand) chiều h và w của x_offset bằng cách nhân với n_x và n_y.
        Lý do: 
        - x_offset có shape (b, c, h, w, N), với N = n_x * n_y là số điểm sampling trong kernel.
        - Để thực hiện tích chập với kernel biến thiên (adaptive kernel), cần sắp xếp lại các giá trị sampling này thành một feature map lớn hơn,
          trong đó mỗi vị trí (h, w) trên feature map gốc sẽ được 'trải' thành một lưới n_x x n_y điểm sampling.
        - Việc reshape này giúp chuyển x_offset từ (b, c, h, w, N) thành (b, c, h * n_x, w * n_y), 
          tức là mỗi điểm (h, w) trên feature map gốc sẽ tương ứng với một vùng (n_x, n_y) trên feature map mới.
        - Điều này cho phép thực hiện các phép toán tích chập tiếp theo như với một feature map thông thường.

        """
        b, c, h, w, N = x_offset.size()
        # Gom các điểm sampling theo từng hàng của kernel (n_y điểm một hàng), rồi ghép lại theo chiều w
        x_offset = torch.cat(
            [x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
            dim=-1
        )
        # Reshape lại thành (b, c, h * n_x, w * n_y)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset