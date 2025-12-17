import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class SimpleEM4EvidenceH_Refine:
    """
    两类根因 δ ∈ {0=EnvFail, 1=AgentFail}
    step 级证据: E1_gui, E2_code, E4_noresp
    case 级证据: agent_testcase_score_x (记为 C), 人工标签 delta_label (0/1)

    特点:
      - 半监督: 有 delta_label 的 case 做硬锚
      - C 通道 case-level 更新 ψ = P(C=1 | δ)
      - 支持各通道权重
      - 反思/ρ 留接口, 默认可以不启用 (你现在 M_reflect=1)

    用法:
      em = SimpleEM4EvidenceH_Refine(...)
      em.fit(df, col_case="test_case_id",
                 col_agent="agent_testcase_score_x",
                 col_delta="delta_label")
      post = em.predict_proba(df_new)
    """

    def __init__(self,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 seed: int = 0,
                 bin_thresh: float = 0.5,
                 # base channel weights
                 w_gui: float = 1.0,
                 w_code: float = 1.0,
                 w_noresp: float = 0.5,
                 # agent channel
                 agent_weight: float = 0.9,
                 # priors
                 a_pi: float = 5.0, b_pi: float = 5.0,  # P_delta prior
                 a_c0: float = 3.0, b_c0: float = 3.0,  # psi EnvFail prior
                 a_c1: float = 3.0, b_c1: float = 3.0,  # psi AgentFail prior
                 theta_floor: float = 0.05,
                 theta_ceil: float = 0.95,
                 pi_floor: float = 0.02,
                 temp: float = 0.8):
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.bin_thresh = float(bin_thresh)

        self.w_gui = float(w_gui)
        self.w_code = float(w_code)
        self.w_no = float(w_noresp)

        self.agent_weight = float(agent_weight)

        self.a_pi, self.b_pi = float(a_pi), float(b_pi)
        self.a_c0, self.b_c0 = float(a_c0), float(b_c0)
        self.a_c1, self.b_c1 = float(a_c1), float(b_c1)

        self.theta_floor = float(theta_floor)
        self.theta_ceil = float(theta_ceil)
        self.pi_floor = float(pi_floor)
        self.temp = float(temp)

        # parameters
        # [EnvFail, AgentFail]
        self.p_delta = np.array([0.5, 0.5], dtype=float)
        # gui/code/noresp
        self.theta = np.full((2, 3), 0.5, dtype=float)
        # P(C=1|EnvFail), P(C=1|AgentFail]
        self.psi = np.array([0.5, 0.5], dtype=float)

        # columns (fit 时覆盖)
        self.col_gui = "E1_gui"
        self.col_code = "E2_code"
        self.col_noresp = "E4_noresp"
        self.col_w = "weight"
        self.col_case = "test_case_id"
        self.col_agent = "agent_testcase_score_x"
        self.col_delta = "delta_label"

    # ---------- utils ----------

    @staticmethod
    def _binarize(x, thresh):
        x = np.asarray(x, float)
        x = np.clip(x, 0.0, 1.0)
        uniq = np.unique(x[~np.isnan(x)])
        if set(uniq).issubset({0.0, 1.0}):
            return x
        return (x >= thresh).astype(float)

    def _extract(self, df: pd.DataFrame):
        # ----- 证据 -----
        Eg = self._binarize(df[self.col_gui], self.bin_thresh)
        Ec = self._binarize(df[self.col_code], self.bin_thresh)

        if self.col_noresp in df.columns:
            En = self._binarize(df[self.col_noresp], self.bin_thresh)
        else:
            En = np.zeros_like(Eg)

        # E[:,0]=gui, E[:,1]=code, E[:,2]=noresp
        E = np.stack([Eg, Ec, En], axis=1)

        # ----- mask: 1 = 忽略该通道 -----
        def _get_mask(col_name):
            if col_name in df.columns:
                m = df[col_name].to_numpy().astype(float)
                return np.where(m >= 0.5, 1.0, 0.0)
            else:
                return np.zeros(Eg.shape[0], float)

        Mg = _get_mask("M_gui")
        Mc = _get_mask("M_code")
        Mn = _get_mask("M_noresp")

        M = np.stack([Mg, Mc, Mn], axis=1)

        # ----- sample 权重 -----
        if self.col_w in df.columns:
            w = np.clip(df[self.col_w].to_numpy().astype(float), 0.0, 10.0)
        else:
            w = np.ones(E.shape[0], float)

        # ----- case id -----
        case_ids = df[self.col_case].astype(str).to_numpy()

        # ----- agent_testcase_score: 行级原始值 (后面按 case 聚合) -----
        C_raw = None
        if self.col_agent in df.columns:
            arr = df[self.col_agent].to_numpy()
            C_raw = np.array([
                np.nan if pd.isna(v) else float(v)
                for v in arr
            ])
            # 压到 {0,1} 或 NaN
            C_raw = np.where(
                np.isnan(C_raw),
                np.nan,
                (C_raw >= 0.5).astype(float)
            )

        # ----- delta_label (半监督 GT，可选) -----
        delta_sup_raw = None
        if self.col_delta in df.columns:
            arr = df[self.col_delta].to_numpy()
            tmp = np.full(len(arr), np.nan)
            for i, v in enumerate(arr):
                if pd.isna(v):
                    continue
                if v in (0, 1):
                    tmp[i] = int(v)
                elif isinstance(v, str):
                    vs = v.strip().lower()
                    if vs.startswith("env"):
                        tmp[i] = 0
                    elif vs.startswith("agent"):
                        tmp[i] = 1
            delta_sup_raw = tmp

        # 关键：现在返回 6 个量，包含 M
        return E, w, M, case_ids, C_raw, delta_sup_raw

    def _init_params(self, E):
        self.p_delta[:] = np.array([0.5, 0.5])
        m = E.mean(axis=0)
        # 让 AgentFail 在 gui/code/noresp 上稍微“更错一点”
        self.theta[0, :] = np.clip(m, 0.2, 0.8)   # Env
        self.theta[1, :] = np.clip(m + 0.15, 0.2, 0.9)  # Agent
        self.psi[:] = np.array([0.4, 0.8])

    # ---------- EM fit ----------

    def fit(self,
            df: pd.DataFrame,
            col_case: str = "test_case_id",
            col_agent: str = "agent_testcase_score_x",
            col_delta: str = "delta_label",
            col_gui: str = "E1_gui",
            col_code: str = "E2_code",
            col_noresp: str = "E4_noresp",
            col_w: str = "weight"):

        # bind column names
        self.col_case = col_case
        self.col_agent = col_agent
        self.col_delta = col_delta
        self.col_gui = col_gui
        self.col_code = col_code
        self.col_noresp = col_noresp
        self.col_w = col_w

        # E, w, case_ids, C_raw, delta_sup_raw = self._extract(df)
        E, w, M, case_ids, C_raw, delta_sup_raw = self._extract(df)
        N = E.shape[0]
        eps = 1e-9

        # case-level index
        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)

        # 聚合 case-level C, delta_sup
        C_case = np.full(K, np.nan)
        delta_case = np.full(K, np.nan)
        for k in range(K):
            idx = (inv_case == k)
            if C_raw is not None:
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    # 这里用最后一个非 nan（或 majority 都可）
                    C_case[k] = vals[-1]
            if delta_sup_raw is not None:
                vals = delta_sup_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    delta_case[k] = vals[-1]

        self._init_params(E)

        ll_prev = -np.inf

        for it in range(self.max_iter):
            T = max(self.temp, 1e-3)

            # ----- E-step (case-level) -----
            # Step 1: 计算每行的 base_log（gui/code/noresp，考虑 mask）
            base_log = np.zeros((N, 2))
            for d in (0, 1):
                p_gui, p_cod, p_no = self.theta[d]
                lg = E[:, 0]*np.log(p_gui+eps) + \
                    (1-E[:, 0])*np.log(1-p_gui+eps)
                lc = E[:, 1]*np.log(p_cod+eps) + \
                    (1-E[:, 1])*np.log(1-p_cod+eps)
                ln = E[:, 2]*np.log(p_no+eps) + (1-E[:, 2])*np.log(1-p_no+eps)
                # 对 mask=1 的位置，把该通道贡献清空
                lg[M[:, 0] == 1] = 0.0
                lc[M[:, 1] == 1] = 0.0
                ln[M[:, 2] == 1] = 0.0
                base_log[:, d] = self.w_gui*lg + self.w_code*lc + self.w_no*ln

            # Step 2: 按 case 求和得到 case_log[k,d]
            case_log = np.zeros((K, 2))
            for k in range(K):
                idx = (inv_case == k)
                # 可选：加权或不加权，这里用加权求和
                case_log[k] = (w[idx, None] * base_log[idx]).sum(axis=0)

            # Step 3: agent 通道只对 case 加一次
            if C_case is not None:
                for k in range(K):
                    if np.isnan(C_case[k]):
                        continue
                    c = C_case[k]
                    for d in (0, 1):
                        psi_d = np.clip(self.psi[d], 1e-4, 1-1e-4)
                        case_log[k, d] += self.agent_weight * (
                            c * np.log(psi_d) + (1 - c) * np.log(1 - psi_d)
                        )

            # Step 4: prior + temp + softmax -> resp_case
            log_num_case = (np.log(self.p_delta + eps)[None, :] + case_log) / T
            m_case = log_num_case.max(axis=1, keepdims=True)
            log_den_case = m_case + \
                np.log(np.exp(log_num_case - m_case).sum(axis=1, keepdims=True) + eps)
            resp_case = np.exp(log_num_case - log_den_case)  # (K, 2)

            # Step 5: 半监督硬锚：对 case 施加
            if not np.all(np.isnan(delta_case)):
                for k in range(K):
                    if np.isnan(delta_case[k]):
                        continue
                    d = int(delta_case[k])  # 0 or 1
                    resp_case[k, :] = 0.0
                    resp_case[k, d] = 1.0

            # Step 6: 广播回行
            resp = resp_case[inv_case]  # (N, 2)

            # ----- M-step -----

            # (1) π with Beta prior
            Nk = (w[:, None] * resp).sum(axis=0)
            self.p_delta = (Nk + np.array([self.a_pi-1, self.b_pi-1])) / \
                           (Nk.sum() + self.a_pi + self.b_pi - 2 + eps)
            self.p_delta = np.clip(
                self.p_delta, self.pi_floor, 1.0 - self.pi_floor)

            # (2) θ for gui/code/noresp（只统计未 mask 的位置）
            for d in (0, 1):
                wk = w * resp[:, d]
                for j in range(3):
                    valid = (M[:, j] == 0)  # 只统计未 mask 的位置
                    wk_valid = wk[valid]
                    E_valid = E[valid, j]
                    ones = (wk_valid * E_valid).sum()
                    den = wk_valid.sum()
                    if den <= 0:
                        p = 0.5
                    else:
                        # 加一点伪计数偏向 2，避免极端
                        num_hat = ones + 0.5
                        den_hat = den + 1.0
                        p = num_hat / (den_hat + eps)
                    self.theta[d, j] = float(
                        np.clip(p, self.theta_floor, self.theta_ceil))

            # (3) 更新 psi (C 通道)，在 case-level 上
            if C_case is not None:
                # 对每个 case 计算该 case 属于 δ 的责任
                Rcase = np.zeros((K, 2))
                for k in range(K):
                    idx = (inv_case == k)
                    if not idx.any():
                        continue
                    wk = w[idx][:, None] * resp[idx, :]
                    Rcase[k, :] = wk.sum(axis=0)
                # EnvFail
                ones0 = 0.0
                den0 = 0.0
                ones1 = 0.0
                den1 = 0.0
                for k in range(K):
                    if np.isnan(C_case[k]):
                        continue
                    c = C_case[k]
                    r0, r1 = Rcase[k, 0], Rcase[k, 1]
                    ones0 += r0 * c
                    den0 += r0
                    ones1 += r1 * c
                    den1 += r1
                if den0 > 0:
                    self.psi[0] = float(
                        np.clip((ones0 + self.a_c0 - 1) /
                                (den0 + self.a_c0 + self.b_c0 - 2 + eps),
                                0.02, 0.98)
                    )
                if den1 > 0:
                    self.psi[1] = float(
                        np.clip((ones1 + self.a_c1 - 1) /
                                (den1 + self.a_c1 + self.b_c1 - 2 + eps),
                                0.02, 0.98)
                    )

            # (4) LL convergence（使用 case-level log-likelihood）
            # 每个 case 的权重：该 case 内所有行的权重之和
            w_case = np.zeros(K)
            for k in range(K):
                idx = (inv_case == k)
                w_case[k] = w[idx].sum()
            avg_ll = float((w_case * log_den_case.squeeze()
                            ).sum() / (w_case.sum() + eps))
            if abs(avg_ll - ll_prev) < self.tol:
                break
            ll_prev = avg_ll

    # ---------- inference ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        返回 step-level P(EnvFail), P(AgentFail)
        内部使用 case-level 聚合证据，然后广播回每行，与 fit 保持一致
        """
        E, w, M, case_ids, C_raw, _ = self._extract(df)
        N = E.shape[0]
        eps = 1e-9

        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)

        # ---- case 级 C，与 fit 同逻辑 ----
        C_case = np.full(K, np.nan)
        if C_raw is not None:
            for k in range(K):
                idx = (inv_case == k)
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    C_case[k] = vals[-1]

        # ---- Step 1: 计算每行的 base_log ----
        base_log = np.zeros((N, 2))
        for d in (0, 1):
            p_gui, p_cod, p_no = self.theta[d]

            lg = E[:, 0]*np.log(p_gui+eps) + (1-E[:, 0])*np.log(1-p_gui+eps)
            lc = E[:, 1]*np.log(p_cod+eps) + (1-E[:, 1])*np.log(1-p_cod+eps)
            ln = E[:, 2]*np.log(p_no+eps) + (1-E[:, 2])*np.log(1-p_no+eps)

            # 对 mask=1 的位置，把该通道贡献清空
            lg[M[:, 0] == 1] = 0.0
            lc[M[:, 1] == 1] = 0.0
            ln[M[:, 2] == 1] = 0.0

            base_log[:, d] = self.w_gui*lg + self.w_code*lc + self.w_no*ln

        # ---- Step 2: 按 case 求和得到 case_log ----
        case_log = np.zeros((K, 2))
        for k in range(K):
            idx = (inv_case == k)
            case_log[k] = (w[idx, None] * base_log[idx]).sum(axis=0)

        # ---- Step 3: agent 通道只对 case 加一次 ----
        if C_case is not None:
            for k in range(K):
                if np.isnan(C_case[k]):
                    continue
                c = C_case[k]
                for d in (0, 1):
                    psi_d = np.clip(self.psi[d], 1e-4, 1-1e-4)
                    case_log[k, d] += self.agent_weight * (
                        c * np.log(psi_d) + (1 - c) * np.log(1 - psi_d)
                    )

        # ---- Step 4: prior + temp + softmax -> resp_case ----
        T = max(self.temp, 1e-3)
        log_num_case = (np.log(self.p_delta + eps)[None, :] + case_log) / T
        m_case = log_num_case.max(axis=1, keepdims=True)
        log_den_case = m_case + \
            np.log(np.exp(log_num_case - m_case).sum(axis=1, keepdims=True) + eps)
        resp_case = np.exp(log_num_case - log_den_case)  # (K, 2)

        # ---- Step 5: 广播回行 ----
        post = resp_case[inv_case]  # (N, 2)
        return post   # [:,0]=P(EnvFail), [:,1]=P(AgentFail)

    def predict_proba_case(self, df: pd.DataFrame):
        """
        返回 case-level posteriors，方便纠偏使用
        Returns:
            uniq_case: (K,) unique case IDs
            resp_case: (K, 2) P(EnvFail), P(AgentFail) for each case
            C_case: (K,) agent original judgment for each case (may contain NaN)
        """
        E, w, M, case_ids, C_raw, _ = self._extract(df)
        N = E.shape[0]
        eps = 1e-9

        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)

        # ---- case 级 C ----
        C_case = np.full(K, np.nan)
        if C_raw is not None:
            for k in range(K):
                idx = (inv_case == k)
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    C_case[k] = vals[-1]

        # ---- Step 1: 计算每行的 base_log ----
        base_log = np.zeros((N, 2))
        for d in (0, 1):
            p_gui, p_cod, p_no = self.theta[d]

            lg = E[:, 0]*np.log(p_gui+eps) + (1-E[:, 0])*np.log(1-p_gui+eps)
            lc = E[:, 1]*np.log(p_cod+eps) + (1-E[:, 1])*np.log(1-p_cod+eps)
            ln = E[:, 2]*np.log(p_no+eps) + (1-E[:, 2])*np.log(1-p_no+eps)

            lg[M[:, 0] == 1] = 0.0
            lc[M[:, 1] == 1] = 0.0
            ln[M[:, 2] == 1] = 0.0

            base_log[:, d] = self.w_gui*lg + self.w_code*lc + self.w_no*ln

        # ---- Step 2: 按 case 求和 ----
        case_log = np.zeros((K, 2))
        for k in range(K):
            idx = (inv_case == k)
            case_log[k] = (w[idx, None] * base_log[idx]).sum(axis=0)

        # ---- Step 3: agent 通道 ----
        if C_case is not None:
            for k in range(K):
                if np.isnan(C_case[k]):
                    continue
                c = C_case[k]
                for d in (0, 1):
                    psi_d = np.clip(self.psi[d], 1e-4, 1-1e-4)
                    case_log[k, d] += self.agent_weight * (
                        c * np.log(psi_d) + (1 - c) * np.log(1 - psi_d)
                    )

        # ---- Step 4: temp + softmax ----
        T = max(self.temp, 1e-3)
        log_num_case = (np.log(self.p_delta + eps)[None, :] + case_log) / T
        m_case = log_num_case.max(axis=1, keepdims=True)
        log_den_case = m_case + \
            np.log(np.exp(log_num_case - m_case).sum(axis=1, keepdims=True) + eps)
        resp_case = np.exp(log_num_case - log_den_case)  # (K, 2)

        return uniq_case, resp_case, C_case

    def get_params(self) -> Dict[str, Any]:
        return {
            "P_delta": {"EnvFail": float(self.p_delta[0]),
                        "AgentFail": float(self.p_delta[1])},
            "theta": {
                "EnvFail":  {"E_gui": float(self.theta[0, 0]),
                             "E_code": float(self.theta[0, 1]),
                             "E_noresp": float(self.theta[0, 2])},
                "AgentFail": {"E_gui": float(self.theta[1, 0]),
                              "E_code": float(self.theta[1, 1]),
                              "E_noresp": float(self.theta[1, 2])},
            },
            "psi": {
                "EnvFail": float(self.psi[0]),
                "AgentFail": float(self.psi[1]),
            }
        }

    def load_params(self, params: Dict[str, Any]):
        """
        从参数字典加载已训练的参数
        """
        self.p_delta[0] = float(params["P_delta"]["EnvFail"])
        self.p_delta[1] = float(params["P_delta"]["AgentFail"])

        self.theta[0, 0] = float(params["theta"]["EnvFail"]["E_gui"])
        self.theta[0, 1] = float(params["theta"]["EnvFail"]["E_code"])
        self.theta[0, 2] = float(params["theta"]["EnvFail"]["E_noresp"])
        self.theta[1, 0] = float(params["theta"]["AgentFail"]["E_gui"])
        self.theta[1, 1] = float(params["theta"]["AgentFail"]["E_code"])
        self.theta[1, 2] = float(params["theta"]["AgentFail"]["E_noresp"])

        self.psi[0] = float(params["psi"]["EnvFail"])
        self.psi[1] = float(params["psi"]["AgentFail"])

    def _score_avg_ll(self, E, w, case_ids, C_raw):
        """
        计算平均 log-likelihood
        参数：
        - E: (N, 3) 证据矩阵 [gui, code, noresp]
        - w: (N,) 权重
        - case_ids: (N,) case ID（用于聚合 C）
        - C_raw: (N,) agent score（可能包含 NaN）
        """
        N = E.shape[0]
        eps = 1e-9

        # 聚合 C_case（和 predict_proba 同逻辑）
        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)
        C_case = np.full(K, np.nan)
        if C_raw is not None:
            for k in range(K):
                idx = (inv_case == k)
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    C_case[k] = vals[-1]

        # base log-likelihood
        base_log = np.zeros((N, 2))
        for d in (0, 1):
            p_gui, p_cod, p_no = self.theta[d]
            lg = E[:, 0]*np.log(p_gui+eps) + (1-E[:, 0])*np.log(1-p_gui+eps)
            lc = E[:, 1]*np.log(p_cod+eps) + (1-E[:, 1])*np.log(1-p_cod+eps)
            ln = E[:, 2]*np.log(p_no+eps) + (1-E[:, 2])*np.log(1-p_no+eps)
            base_log[:, d] = self.w_gui*lg + self.w_code*lc + self.w_no*ln

        # agent log-likelihood
        agent_log = np.zeros((N, 2))
        if C_case is not None:
            C_row = C_case[inv_case]
            mask = ~np.isnan(C_row)
            if mask.any():
                C_obs = C_row[mask]
                for d in (0, 1):
                    psi_d = np.clip(self.psi[d], 1e-4, 1-1e-4)
                    lr = C_obs*np.log(psi_d) + (1-C_obs)*np.log(1-psi_d)
                    agent_log[mask, d] = self.agent_weight * lr

        # 计算 log-likelihood (使用与 fit/predict 一致的温度)
        T = max(self.temp, 1e-3)
        log_num = (np.log(self.p_delta+eps)
                   [None, :] + base_log + agent_log) / T
        m = log_num.max(axis=1, keepdims=True)
        log_den = m + \
            np.log(np.exp(log_num-m).sum(axis=1, keepdims=True) + eps)

        return float((w * log_den.squeeze()).sum() / (w.sum() + eps))


def analyze_flips(val_df: pd.DataFrame, out_dir: Optional[str] = None):
    """区分两类子集: 原判断正确 vs 错误，查看flip比例"""
    from pathlib import Path

    df = val_df.copy()
    df["is_correct_before"] = (df["human_gt"] == df["agent_original"])
    df["is_correct_after"] = (df["human_gt"] == df["corrected_label"])
    df["is_flipped"] = df["agent_original"] != df["corrected_label"]

    # subset A: 原判断正确
    subset_A = df[df["is_correct_before"]]
    misflipped = subset_A[subset_A["is_flipped"]]
    if len(subset_A) > 0:
        print(f"[Subset A] 原判断正确: {len(subset_A)} cases, "
              f"被误翻 {len(misflipped)} ({len(misflipped)/len(subset_A):.2%})")

        # 进一步区分原始标签是0和原始标签是1的误翻情况
        misflipped_0 = misflipped[misflipped["agent_original"] == 0]
        misflipped_1 = misflipped[misflipped["agent_original"] == 1]
        subset_A_0 = subset_A[subset_A["agent_original"] == 0]
        subset_A_1 = subset_A[subset_A["agent_original"] == 1]

        if len(subset_A_0) > 0:
            print(f"  - 原始标签=0: {len(subset_A_0)} cases, "
                  f"被误翻 {len(misflipped_0)} ({len(misflipped_0)/len(subset_A_0):.2%})")
        if len(subset_A_1) > 0:
            print(f"  - 原始标签=1: {len(subset_A_1)} cases, "
                  f"被误翻 {len(misflipped_1)} ({len(misflipped_1)/len(subset_A_1):.2%})")

        # 保存数据
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # 保存被误翻的数据（按原始标签分组）
            if len(misflipped_0) > 0:
                misflipped_0.to_csv(
                    out_path / "misflipped_original_0.csv", index=False)
                print(
                    f"  已保存原始标签=0的误翻数据到: {out_path / 'misflipped_original_0.csv'}")
            if len(misflipped_1) > 0:
                misflipped_1.to_csv(
                    out_path / "misflipped_original_1.csv", index=False)
                print(
                    f"  已保存原始标签=1的误翻数据到: {out_path / 'misflipped_original_1.csv'}")

            # 保存原判断正确的所有数据（按原始标签分组）
            if len(subset_A_0) > 0:
                subset_A_0.to_csv(
                    out_path / "subset_A_original_0.csv", index=False)
                print(
                    f"  已保存原始标签=0的原判断正确数据到: {out_path / 'subset_A_original_0.csv'}")
            if len(subset_A_1) > 0:
                subset_A_1.to_csv(
                    out_path / "subset_A_original_1.csv", index=False)
                print(
                    f"  已保存原始标签=1的原判断正确数据到: {out_path / 'subset_A_original_1.csv'}")
    else:
        print(
            f"[Subset A] 原判断正确: {len(subset_A)} cases, 被误翻 {len(misflipped)}")

    # subset B: 原判断错误
    subset_B = df[~df["is_correct_before"]]
    corrected = subset_B[subset_B["is_correct_after"]]
    not_corrected = subset_B[~subset_B["is_correct_after"]]
    if len(subset_B) > 0:
        print(f"[Subset B] 原判断错误: {len(subset_B)} cases, "
              f"被成功纠正 {len(corrected)} ({len(corrected)/len(subset_B):.2%})")

        # 进一步区分原始标签是0和原始标签是1的情况
        subset_B_0 = subset_B[subset_B["agent_original"] == 0]
        subset_B_1 = subset_B[subset_B["agent_original"] == 1]
        corrected_0 = corrected[corrected["agent_original"] == 0]
        corrected_1 = corrected[corrected["agent_original"] == 1]

        if len(subset_B_0) > 0:
            print(f"  - 原始标签=0: {len(subset_B_0)} cases, "
                  f"被成功纠正 {len(corrected_0)} ({len(corrected_0)/len(subset_B_0):.2%})")
        if len(subset_B_1) > 0:
            print(f"  - 原始标签=1: {len(subset_B_1)} cases, "
                  f"被成功纠正 {len(corrected_1)} ({len(corrected_1)/len(subset_B_1):.2%})")

        # 保存数据
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # 保存原判断错误的所有数据
            subset_B.to_csv(out_path / "subset_B.csv", index=False)
            print(f"  已保存原判断错误数据到: {out_path / 'subset_B.csv'}")

            # 保存被成功纠正的数据（按原始标签分组）
            if len(corrected_0) > 0:
                corrected_0.to_csv(
                    out_path / "subset_B_corrected_original_0.csv", index=False)
                print(
                    f"  已保存原始标签=0的成功纠正数据到: {out_path / 'subset_B_corrected_original_0.csv'}")
            if len(corrected_1) > 0:
                corrected_1.to_csv(
                    out_path / "subset_B_corrected_original_1.csv", index=False)
                print(
                    f"  已保存原始标签=1的成功纠正数据到: {out_path / 'subset_B_corrected_original_1.csv'}")

            # 保存原判断错误但未被纠正的数据
            if len(not_corrected) > 0:
                not_corrected.to_csv(
                    out_path / "subset_B_not_corrected.csv", index=False)
                print(
                    f"  已保存原判断错误但未被纠正数据到: {out_path / 'subset_B_not_corrected.csv'}")
    else:
        print(
            f"[Subset B] 原判断错误: {len(subset_B)} cases, 被成功纠正 {len(corrected)}")

    # 再打印最终 confusion
    print("\n=== Confusion Matrix (GT vs Corrected) ===")
    print(pd.crosstab(df["human_gt"], df["corrected_label"],
                      rownames=["GT"], colnames=["Corrected"]))


def confusion_matrix(val_df: pd.DataFrame):
    """统计人类标注 vs 矫正后结果"""
    conf_matrix = pd.crosstab(val_df["human_gt"], val_df["corrected_label"],
                              rownames=["GT"], colnames=["Corrected"])
    print("=== Confusion Matrix (GT vs Corrected) ===")
    print(conf_matrix)
    acc = (val_df["human_gt"] == val_df["corrected_label"]).mean()
    print(f"Accuracy after correction: {acc:.3f}")
    return conf_matrix


def correct_agent_judgment(df: pd.DataFrame,
                           em: SimpleEM4EvidenceH_Refine,
                           margin_agentfail: float = 0.8,
                           tau_envfail_high: float = 0.7,
                           tau_agentfail_high: float = 0.95,
                           tau_agentfail_floor: float = 0.65,
                           tau_support: float = 0.0,
                           tau_support_pos: float = None,
                           tau_support_neg: float = None,
                           min_neg_channels: int = 0,
                           col_case: str = "test_case_id",
                           col_agent: str = "agent_testcase_score_x",
                           out_dir: Optional[str] = None):
    """
    对每个 case：
      - 直接使用 case-level posterior (resp_case)
      - 与 agent 原判 (C_case) 结合，给出纠偏动作

    0→1 (FAIL→PASS) 纠正规则（以证据为主）：
      主门槛：evidence_support >= tau_support_pos (硬证据，最靠谱)
      低保底：P_case_AgentFail >= tau_agentfail_floor (0.65/0.7)
      保留：logodds >= margin_agentfail
      三者都满足才翻转

    1→0 (PASS→FAIL) 纠正规则（更严格）：
      tau_agentfail_high: 用于 1→0 判断中的概率门槛
      tau_envfail_high: P_case_EnvFail 需要达到的门槛
      tau_support_neg: 证据支持负向门槛（更严格）
      min_neg_channels: 多通道一致性要求

    新增"证据差异门槛" tau_support：
      support_sum = Σ_step Σ_channel valid * w_ch * [log P(E|AgentFail) - log P(E|EnvFail)]
      support_avg = support_sum / #valid_evidence  (归一化，避免"步数堆票")
      避免仅靠 prior/psi 或单一弱通道把 posterior 推爆

    非对称纠偏规则（重要！）：
      - tau_support_pos: 用于 0→1 (FAIL→PASS) 的翻转门槛（硬证据主门槛）
      - tau_support_neg: 用于 1→0 (PASS→FAIL) 的翻转门槛（更严格）
      如果不指定，则回退到对称的 tau_support

    多通道一致性（对 1→0 额外保护）：
      - min_neg_channels: 至少需要多少个通道的证据支持 EnvFail
        (即 llr_per_channel < 0 的有效通道数 >= min_neg_channels)
    """
    # 处理非对称门槛参数
    if tau_support_pos is None:
        tau_support_pos = tau_support
    if tau_support_neg is None:
        tau_support_neg = tau_support
    eps = 1e-9

    # 直接获取 case-level posteriors
    uniq_case, resp_case, C_case_arr = em.predict_proba_case(df)

    # ---- 计算每个 case 的证据支持度 (evidence support) ----
    # support_sum = Σ_step Σ_channel (1-M) * w_ch * [log P(E|AgentFail) - log P(E|EnvFail)]
    # support_avg = support_sum / #valid_evidence  (归一化，避免步数堆票)
    E, w, M, case_ids, _, _ = em._extract(df)
    N = E.shape[0]
    _, inv_case = np.unique(case_ids, return_inverse=True)
    K = len(uniq_case)

    # 计算每行每通道的 log-likelihood 差 (AgentFail - EnvFail)
    # llr[i, j] = log P(E[i,j] | AgentFail) - log P(E[i,j] | EnvFail)
    channel_weights = np.array([em.w_gui, em.w_code, em.w_no])
    llr_per_channel = np.zeros((N, 3))  # gui, code, noresp
    for j in range(3):
        p_env = em.theta[0, j]
        p_agent = em.theta[1, j]
        # log P(E|AgentFail) - log P(E|EnvFail)
        ll_agent = E[:, j] * np.log(p_agent + eps) + \
            (1 - E[:, j]) * np.log(1 - p_agent + eps)
        ll_env = E[:, j] * np.log(p_env + eps) + \
            (1 - E[:, j]) * np.log(1 - p_env + eps)
        llr_per_channel[:, j] = ll_agent - ll_env

    # 加权并按 mask 过滤：valid = (1 - M)
    valid = 1.0 - M  # (N, 3)
    weighted_llr = valid * channel_weights[None, :] * llr_per_channel  # (N, 3)
    step_support = weighted_llr.sum(axis=1)  # (N,) 每行的总支持度
    step_valid_count = valid.sum(axis=1)  # (N,) 每行的有效证据通道数

    # 按 case 聚合，归一化避免"步数堆票"
    # support_avg = support_sum / #valid_evidence
    support_case = np.zeros(K)
    # 多通道一致性：统计每个 case 有多少个有效通道的 llr < 0 (支持 EnvFail)
    neg_channel_count_case = np.zeros(K, dtype=int)
    # 每个通道的平均 LLR（用于诊断 misflip 原因）
    avg_llr_per_channel_case = np.zeros((K, 3))  # [gui, code, noresp]

    for k in range(K):
        idx = (inv_case == k)
        support_sum = step_support[idx].sum()
        valid_evidence_count = step_valid_count[idx].sum()  # 该 case 的总有效证据数
        if valid_evidence_count > 0:
            support_case[k] = support_sum / valid_evidence_count  # 归一化
        else:
            support_case[k] = 0.0

        # 统计强负通道数（用于多通道一致性检查）
        # 对该 case 的所有 step，统计每个有效通道的平均 llr，看有多少个通道 < 0
        case_valid = valid[idx]  # (n_steps, 3)
        case_llr = llr_per_channel[idx]  # (n_steps, 3)
        for j in range(3):
            valid_count_j = case_valid[:, j].sum()
            if valid_count_j > 0:
                avg_llr_j = (case_valid[:, j] *
                             case_llr[:, j]).sum() / valid_count_j
                avg_llr_per_channel_case[k, j] = avg_llr_j  # 保存每个通道的平均 LLR
                if avg_llr_j < 0:  # 该通道平均支持 EnvFail
                    neg_channel_count_case[k] += 1
            else:
                avg_llr_per_channel_case[k, j] = np.nan  # 无有效证据

    # 获取每个 case 的 GT (phi)
    gt_map = {}
    for cid, g in df.groupby(col_case):
        gt_vals = g["phi"].dropna().values
        if len(gt_vals):
            gt_map[cid] = gt_vals[-1]
        else:
            gt_map[cid] = np.nan

    rows = []
    for k in range(K):
        cid = uniq_case[k]
        P_case_EnvFail = resp_case[k, 0]
        P_case_AgentFail = resp_case[k, 1]
        C_case = C_case_arr[k]
        case_support = support_case[k]  # 该 case 的证据支持度

        # agent 原判
        if np.isnan(C_case):
            C_case_int = None
        else:
            C_case_int = int(C_case)

        gt = gt_map.get(cid, np.nan)

        action = "keep_AgentJudge"
        corrected = C_case_int

        # 获取该 case 的强负通道数
        neg_channels = neg_channel_count_case[k]

        if C_case_int is not None:
            # 场景1: agent 判 FAIL (0) → 可能翻转为 PASS (1)
            # 以证据为主的三门槛规则：
            #   主门槛：evidence_support >= tau_support_pos (硬证据，最靠谱)
            #   低保底：P_case_AgentFail >= tau_agentfail_floor (0.65/0.7)
            #   保留：logodds >= margin_agentfail
            if C_case_int == 0:
                # log-odds = log(P_AgentFail) - log(P_EnvFail)
                logodds = np.log(P_case_AgentFail + eps) - \
                    np.log(P_case_EnvFail + eps)
                # 0→1 翻转：主门槛是证据，P_case_AgentFail 只是低保底
                if (logodds >= margin_agentfail and
                    P_case_AgentFail >= tau_agentfail_floor and
                        case_support >= tau_support_pos):
                    corrected = 1  # 认定 AgentFail，翻转
                    action = "flip_to_AgentFail"
                else:
                    corrected = 0  # 环境问题或不确定，维持
                    action = "keep_EnvFail"

            # 场景2: agent 判 PASS (1) → 可能翻转为 FAIL (0)
            # 1→0 翻转代价更大，需要更严格的条件：
            #   - P_case_EnvFail >= tau_envfail_high
            #   - case_support <= -tau_support_neg（更严格的负向支持）
            #   - 多通道一致性：至少 min_neg_channels 个通道支持 EnvFail
            elif C_case_int == 1:
                # 对于 EnvFail 方向的翻转，support 应为负值（支持 EnvFail）
                # 使用 -tau_support_neg 作为门槛（更严格）
                support_ok = (case_support <= -tau_support_neg)
                channel_ok = (neg_channels >= min_neg_channels)

                if P_case_EnvFail >= tau_envfail_high and support_ok and channel_ok:
                    corrected = 0
                    action = "flip_to_EnvFail"

        # 获取该 case 每个通道的平均 LLR
        avg_llr_gui = avg_llr_per_channel_case[k, 0]
        avg_llr_code = avg_llr_per_channel_case[k, 1]
        avg_llr_noresp = avg_llr_per_channel_case[k, 2]

        rows.append(dict(
            case_id=cid,
            human_gt=gt,
            agent_original=C_case_int,
            P_case_EnvFail=float(P_case_EnvFail),
            P_case_AgentFail=float(P_case_AgentFail),
            logodds=float(np.log(P_case_AgentFail + eps) -
                          np.log(P_case_EnvFail + eps)),
            evidence_support=float(case_support),  # 证据支持度
            neg_channels=int(neg_channels),  # 支持 EnvFail 的通道数
            avg_llr_gui=float(avg_llr_gui) if not np.isnan(
                avg_llr_gui) else None,
            avg_llr_code=float(avg_llr_code) if not np.isnan(
                avg_llr_code) else None,
            avg_llr_noresp=float(avg_llr_noresp) if not np.isnan(
                avg_llr_noresp) else None,
            corrected_label=corrected,
            action=action
        ))

    val_df = pd.DataFrame(rows).sort_values("case_id")
    acc_original = val_df["agent_original"] == val_df["human_gt"]
    print(f"Accuracy: {acc_original.mean()}")

    acc_correct = val_df["corrected_label"] == val_df["human_gt"]
    print(f"Accuracy: {acc_correct.mean()}")
    analyze_flips(val_df, out_dir=out_dir)
    confusion_matrix(val_df)
    return val_df
