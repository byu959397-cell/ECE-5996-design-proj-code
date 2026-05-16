import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthoHistLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank=16, alpha=32, dropout=0.0, mt_rank=8):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.mt_rank = mt_rank
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self._init_active_fresh()

        # pool stores alternating (basis [d_out, r], coeff [r, d_in]) as frozen params
        self.basis_pool = nn.ParameterList()
        self._n_pool = 0

        # current task's mixing matrix M_t = U_mt @ V_mt.T
        self.mt_U: Optional[nn.Parameter] = None
        self.mt_V: Optional[nn.Parameter] = None

        # keep archived M_t's for analysis   
        # Store the Mt for each task (does not count as a parameter, the true parameters consist solely of the basis pool, the active zone, and the current Mt, not including all Mts)
        self._frozen_mt: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []

        self._hook = self.register_full_backward_hook(self._ortho_grad_hook)

    def _init_active_fresh(self):
        # standard LoRA init for task 1 kaiming: suitable for deep neural networks
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def init_active_projected(self):
      if self._n_pool == 0:
          nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
          nn.init.zeros_(self.lora_B)
          return

      device = self.lora_B.device
      dtype = self.lora_B.dtype
      pool_B = self._cat_pool_basis().float()
      # proj为0的情况（历史pool把所有方向都占了）
      # 高斯：随机采样方向 QR后都一样
      B_rand = torch.randn(self.out_features, self.rank, device=device).float()
      # QR
      B_proj = B_rand - pool_B @ (pool_B.T @ B_rand)
      # mode reduced？
      q, _ = torch.linalg.qr(B_proj, mode="reduced")

      with torch.no_grad():
          self.lora_B.copy_(q.to(dtype=dtype, device=device))
      
      nn.init.zeros_(self.lora_A) 


    def init_mt(self):
      r_total = self._n_pool * self.rank

      if r_total == 0:
          self.mt_U = None
          self.mt_V = None
          return

      device = self.lora_B.device
      dtype  = self.lora_B.dtype

      U_new = torch.empty(
          r_total,
          self.mt_rank,
          device=device,
          dtype=dtype,
      )

      V_new = torch.empty(
          r_total,
          self.mt_rank,
          device=device,
          dtype=dtype,
      )
      # 小初始化：平滑梯度更新 避免学A不学B
      nn.init.normal_(U_new, mean=0.0, std=0.001)
      nn.init.normal_(V_new, mean=0.0, std=0.001)

      self.mt_U = nn.Parameter(U_new)
      self.mt_V = nn.Parameter(V_new)

    # check
    def forward(self, x):
        base = self.base_layer(x)
        x_d = self.dropout(x)

        active = F.linear(F.linear(x_d, self.lora_A), self.lora_B)

        pool_out = x.new_zeros(active.shape)
        if self._n_pool > 0:
            B_pool = self._cat_pool_basis()   # [d_out, r_total]
            C_pool = self._cat_pool_coeff()   # [r_total, d_in]
            mid = F.linear(x_d, C_pool)       # [..., r_total]

            if self.mt_U is not None:
                # apply (I + U_mt V_mt^T) to mid
                mid = mid + (mid @ self.mt_U) @ self.mt_V.T

            pool_out = F.linear(mid, B_pool)

        return base + self.scaling * (active + pool_out)

    def _cat_pool_basis(self):
        return torch.cat([self.basis_pool[2 * i] for i in range(self._n_pool)], dim=1)

    def _cat_pool_coeff(self):
        return torch.cat([self.basis_pool[2 * i + 1] for i in range(self._n_pool)], dim=0)

    def commit_to_pool(self):
        B_trained = self.lora_B.detach().clone().float()   # [d_out, r]
        A_trained = self.lora_A.detach().clone().float()   # [r, d_in]

        # mode="reduced" Q: [d_out, r], R: [r, r]
        Q, R = torch.linalg.qr(B_trained, mode="reduced")

        # coeff = R @ A, Q @ (R@A) = B @ A
        RA = R @ A_trained  # [r, d_in]

        dtype = self.lora_B.dtype
        device = self.lora_B.device

        # commit的时候要设置为false!
        self.basis_pool.append(
            nn.Parameter(Q.to(dtype=dtype, device=device), requires_grad=False)
        )
        self.basis_pool.append(
            nn.Parameter(RA.to(dtype=dtype, device=device), requires_grad=False)
        )
        self._n_pool += 1

    def archive_and_freeze_mt(self):
        U = self.mt_U.detach().clone() if self.mt_U is not None else None
        V = self.mt_V.detach().clone() if self.mt_V is not None else None
        self._frozen_mt.append((U, V))
        # self.mt_U:当前mt  
        if self.mt_U is not None:
            self.mt_U.requires_grad_(False)
            self.mt_V.requires_grad_(False)

    def zero_active_zone(self):
        with torch.no_grad():
            self.lora_A.zero_()
            self.lora_B.zero_()

    @torch.no_grad()
    def _ortho_grad_hook(self, module, grad_input, grad_output):
        # project grad_B away from all pool bases to enforce inter-task orthogonality
        if self.lora_B.grad is None or self._n_pool == 0:
            return
        g = self.lora_B.grad.float()
        for i in range(self._n_pool):
            b = self.basis_pool[2 * i].detach().float()
            g = g - b @ (b.T @ g)
        self.lora_B.grad.copy_(g.to(dtype=self.lora_B.grad.dtype))

    # 
    def mt_block_norms(self):
        # Frobenius norm of each row-block of M_t; block i corresponds to historical task i+1
        if self.mt_U is None or self._n_pool == 0:
            return None
        M = (self.mt_U @ self.mt_V.T).detach().float()
        # row represents reuse
        return [
            M[i * self.rank:(i + 1) * self.rank, :].norm().item()
            for i in range(self._n_pool)
        ]


class OrthoHistManager:
    def __init__(self, model, target_modules, rank=16, alpha=32, dropout=0.0, mt_rank=8, device=None):
        self.model = model
        self.target_modules = tuple(target_modules)
        self.rank = rank
        self.alpha = alpha
        self.mt_rank = mt_rank
        self.device = device or next(model.parameters()).device
        self.current_task_id = 0
        self.layers: Dict[str, OrthoHistLoRALinear] = {}
    # 1.
    def inject(self):
        dtype = next(self.model.parameters()).dtype
        replaced = {}
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(t) for t in self.target_modules):
                continue
            parent_name, child_name = name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)
            wrapper = OrthoHistLoRALinear(
                module, rank=self.rank, alpha=self.alpha, mt_rank=self.mt_rank
            ).to(device=self.device, dtype=dtype)
            setattr(parent, child_name, wrapper)
            replaced[name] = wrapper

        if not replaced:
            raise RuntimeError(f"No target modules found. target_modules={self.target_modules}")
        self.layers = replaced

    # 2. initialization
    def prepare_for_task(self, task_id):
        self.current_task_id = task_id
        for name, layer in self.layers.items():
            if task_id == 1:
                layer._init_active_fresh()
            else:
                layer.init_active_projected()
            layer.init_mt()
        self._set_trainable()

    # 3. set requires grad
    def _set_trainable(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        for layer in self.layers.values():
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)
            if layer.mt_U is not None:
                layer.mt_U.requires_grad_(True)
                layer.mt_V.requires_grad_(True)
    # 4.
    def trainable_parameters(self):
        params, seen = [], set()
        for layer in self.layers.values():
            for p in [layer.lora_A, layer.lora_B, layer.mt_U, layer.mt_V]:
              # id(p): parameter's address
                if p is not None and p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params
    
    # 5
    def commit_task(self):
        for name, layer in self.layers.items():
            # commit B and A
            layer.commit_to_pool()
            # commit M_t
            layer.archive_and_freeze_mt()
            layer.zero_active_zone()

    def n_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def summary(self):
        n_pool = next(iter(self.layers.values()))._n_pool
        return {
            "method": "ortho_hist",
            "task_id": self.current_task_id,
            "n_pool": n_pool,
            "r_total": n_pool * self.rank,
            "mt_rank": self.mt_rank,
            "trainable_params": self.n_trainable_params(),
        }
        
    # 3 收集每个任务每一层的Mt 最后复制 commit  先collect值 最后冻结
    def collect_reuse_analysis(self, task_name):
        result = {"task_name": task_name, "layers": {}}
        for name, layer in self.layers.items():
            norms = layer.mt_block_norms()
            result["layers"][name] = norms if norms is not None else []
        return result
