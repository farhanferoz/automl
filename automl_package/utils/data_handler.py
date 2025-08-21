from sklearn.preprocessing import StandardScaler


class DataHandler:
    def __init__(self, scale_x=True, scale_y=True):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.x_scaler = StandardScaler() if scale_x else None
        self.y_scaler = StandardScaler() if scale_y else None

    def fit_transform(self, x, y):
        x_scaled = self.x_scaler.fit_transform(x) if self.scale_x else x
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten() if self.scale_y else y
        return x_scaled, y_scaled

    def inverse_transform_y(self, y_pred_scaled):
        if not self.scale_y:
            return y_pred_scaled
        return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
