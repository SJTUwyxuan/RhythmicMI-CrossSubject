import torch


class SSMRRTDCA:
    """
    GPU-accelerated TDCA implementation.

    Training input:
        eeg: Tensor, shape (samples, channels, block, class)
    """

    def __init__(
        self,
        fs,
        subspace_num,
        delay_num,
        freq_vec,
        harmonic,
        wind_sec_max=5,
        device=None
    ):
        self.fs = fs
        self.subspace_num = subspace_num
        self.delay_num = delay_num
        self.freq_vec = freq_vec
        self.harmonic = harmonic
        self.wind_sec_max = wind_sec_max
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Learned components
        self.ref_mat_all = None
        self.template_all = None
        self.weight_all = None

    def train(self, eeg: torch.Tensor):
        """
        Train TDCA model.

        Args:
            eeg: Tensor (samples, channels, block, class)
        """
        # Reference matrix: (wind_sec_max*fs, 2*harmonic, class)
        self.ref_mat_all = self._build_ref().to(self.device)

        eeg = eeg.to(self.device)
        S, C, B, K = eeg.shape

        # Templates and spatial weights
        self.template_all = self._update_template(eeg)   # (delay*C, 2*S, K)
        self.weight_all = self._update_weight(eeg)       # (delay*C, subspace_num)

        return self

    def detection(self, epoch: torch.Tensor):
        """
        TDCA detection for a single epoch.

        Args:
            epoch: Tensor (samples, channels)

        Returns:
            pred: int in [1..K]
            rho:  Tensor (K,)
        """
        epoch = epoch.to(self.device)  # (S, C)
        S, C = epoch.shape
        K = self.ref_mat_all.shape[-1]

        W = self.weight_all  # (delay*C, subspace_num)
        TDCAR = torch.zeros(K, device=self.device)

        # Delay embedding (one-shot)
        aug = []
        for d in range(1, self.delay_num + 1):
            pad = torch.zeros(C, d - 1, device=self.device)
            part = torch.cat([epoch.T[:, d - 1:], pad], dim=1)
            aug.append(part)
        aug = torch.cat(aug, dim=0)  # (delay*C, S)

        # Reference projection and correlation scoring
        for j in range(K):
            Ref = self.ref_mat_all[:S, :, j]  # (S, 2*harmonic)
            Q, _ = torch.linalg.qr(Ref)       # (S, 2*harmonic)
            P = Q @ Q.T                       # (S, S)

            aug_P = aug @ P
            Xb = torch.cat([aug, aug_P], dim=1)  # (delay*C, 2*S)

            WXb = W.T @ Xb                        # (subspace, 2*S)
            temp = self.template_all[:, :, j]     # (delay*C, 2*S)
            Wt = W.T @ temp                       # (subspace, 2*S)

            # Pearson correlation
            v1 = WXb.flatten()
            v2 = Wt.flatten()
            vx = v1 - v1.mean()
            vy = v2 - v2.mean()
            rho = (vx * vy).sum() / torch.sqrt((vx ** 2).sum() * (vy ** 2).sum() + 1e-8)
            TDCAR[j] = rho

        pred = torch.argmax(TDCAR).item() + 1
        return pred, TDCAR

    def _build_ref(self):
        """
        Build sinusoidal reference matrix for all classes/frequencies.

        Returns:
            ref: Tensor (S, 2*harmonic, K), where S = wind_sec_max*fs and K=len(freq_vec)
        """
        S = self.wind_sec_max * self.fs
        K = len(self.freq_vec)

        ref = torch.zeros(S, 2 * self.harmonic, K, device=self.device)
        t = torch.arange(0, S, device=self.device) / self.fs

        for i, f in enumerate(self.freq_vec):
            f_vec = torch.arange(1, self.harmonic + 1, device=self.device) * f
            omega = 2 * torch.pi * f_vec.unsqueeze(0) * t.unsqueeze(1)  # (S, harmonic)
            ref[:, :, i] = torch.cat([torch.sin(omega), torch.cos(omega)], dim=1)

        return ref

    def _update_template(self, eeg):
        """
        Estimate class templates (augmented and projected).
        """
        Xa, Xa_t = self._augment(eeg)
        return Xa_t  # (delay*C, 2*S, K)

    def _update_weight(self, eeg):
        """
        Estimate spatial filters via generalized eigen-decomposition of (Sw^-1 Sb).
        """
        Xa, Xa_t = self._augment(eeg)

        M, N, B, K = Xa.shape
        Sw = torch.zeros(M, M, device=self.device)
        Sb = torch.zeros(M, M, device=self.device)

        overall_mean = Xa.mean(dim=(2, 3))

        for c in range(K):
            mean_c = Xa_t[:, :, c]

            for b in range(B):
                Xb = Xa[:, :, b, c]
                if B == 1:
                    X_tmp = Xb
                else:
                    # Subtract the full class template matrix
                    X_tmp = Xb - Xa_t[:, :, c]
                Sw += X_tmp @ X_tmp.T / B

            tmp = mean_c - overall_mean
            Sb += tmp @ tmp.T / K

        eigvals, eigvecs = torch.linalg.eig(torch.linalg.pinv(Sw) @ Sb)
        idx = torch.argsort(eigvals.real, descending=True)
        U = eigvecs[:, idx[:self.subspace_num]].real

        return U  # (delay*C, subspace_num)

    def _augment(self, eeg):
        """
        Build delayed augmentation and reference-projected features.

        Returns:
            Xa:   (M, 2*S, B, K)
            Xa_t: (M, 2*S, K)  class-wise template (mean over blocks)
        """
        S, C, B, K = eeg.shape
        M = self.delay_num * C

        Xa = torch.zeros(M, 2 * S, B, K, device=self.device)
        Xa_t = torch.zeros(M, 2 * S, K, device=self.device)

        for c in range(K):
            Ref = self.ref_mat_all[:S, :, c]
            Q, _ = torch.linalg.qr(Ref)
            P = Q @ Q.T

            for b in range(B):
                parts = []
                for d in range(1, self.delay_num + 1):
                    X = eeg[d - 1:, :, b, c].transpose(0, 1)  # (C, S-d+1)
                    pad = torch.zeros(C, d - 1, device=self.device)
                    parts.append(torch.cat([X, pad], dim=1))

                Xaug = torch.cat(parts, dim=0)  # (M, S)

                Xa[:, :S, b, c] = Xaug
                Xa[:, S:, b, c] = Xaug @ P

            Xa_t[:, :, c] = Xa[:, :, :, c].mean(dim=2)

        return Xa, Xa_t
