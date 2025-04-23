import numpy as np
import csp as CSP


def compute_one_vs_rest_csp(X, y, n_classes=7, n_filters_per_class=1):
    """
    X: np.ndarray, shape [N, C, T]
    y: np.ndarray, shape [N], class labels (0 to 6)
    Returns:
        csp_filters: shape [n_classes * n_filters_per_class, C]
    """
    all_filters = []
    for class_label in range(n_classes):
        # 1-vs-rest mask
        y_binary = np.where(y == class_label, 1, 0)
        
        # Only keep class vs rest data
        idx = np.where(np.isin(y, [class_label]))[0]
        rest_idx = np.where(y != class_label)[0]

        X_class = X[idx]
        X_rest = X[rest_idx]

        X_bin = np.concatenate([X_class, X_rest])
        y_bin = np.concatenate([np.ones(len(X_class)), np.zeros(len(X_rest))])

        # Fit binary CSP
        csp = CSP(n_components=n_filters_per_class)
        csp.fit(X_bin, y_bin)

        # Collect the top filters (shape: [n_filters_per_class, C])
        filters = csp.filters[:n_filters_per_class]
        all_filters.append(filters)

    # Stack all filters: shape [n_classes * n_filters_per_class, C]
    csp_filters = np.vstack(all_filters)
    return csp_filters