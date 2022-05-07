def torch_sle_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Calculate the Squared Log Error loss."""
    return 1/2 * (torch.log1p(y_pred) - torch.log1p(y_true))**2
